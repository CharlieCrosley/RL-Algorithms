"""

Deep Q-Network (DQN) implementation.

"""

import math
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from models.shared.base_model import BaseModel
from models.shared.data import Transition, ReplayMemory
from models.shared.core import DeterministicPolicy


class DQN(BaseModel):

    def __init__(self, config, env):
        super().__init__(config, env)

        self.policy = DeterministicPolicy(self.n_observations, self.n_actions, hidden_layers=config.policy_hidden_n_layers, 
                                        hidden_sizes=self.config.policy_hidden_sizes, hidden_activation='tanh', 
                                        frame_stack=self.config.frame_stack, bias=self.config.bias)
        
        self.target_policy = DeterministicPolicy(self.n_observations, self.n_actions, hidden_layers=config.policy_hidden_n_layers, 
                                        hidden_sizes=self.config.policy_hidden_sizes, hidden_activation='tanh', 
                                        frame_stack=self.config.frame_stack, bias=self.config.bias)
        
        self.target_policy.load_state_dict(self.policy.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.policy_lr) 

        self.memory = ReplayMemory(self.config.memory_size)
        
        self.best_loss = -float('inf')
    
    def compute_loss(self, states, actions, next_states, rewards, terminal):
        state_action_values = self.policy(states).gather(1, actions)

        with torch.no_grad():
            next_state_values = self.target_policy(next_states).max(1)[0]
      
        # Compute the expected Q values
        expected_state_action_values = rewards + self.config.gamma * next_state_values * terminal

        policy_loss = F.smooth_l1_loss(state_action_values.view(-1), expected_state_action_values)
        return policy_loss

    def optimize(self):
        transitions = self.memory.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))
        
        states = torch.stack(batch.state).to(self.device, torch.float32)
        actions = torch.stack(batch.action).to(self.device, torch.long).view(-1, 1)
        next_states = torch.stack(batch.next_state).to(self.device, torch.float32)
        rewards = torch.tensor(batch.reward).to(self.device, torch.float32)
        terminal = torch.tensor(batch.terminal).to(self.device)

        policy_loss = self.compute_loss(states, actions, next_states, rewards, terminal)
  
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # gradient clipping so that it doesnt blow up
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.policy_optimizer.step()

        target_net_state_dict = self.target_policy.state_dict()
        policy_net_state_dict = self.policy.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.config.tau + target_net_state_dict[key]*(1-self.config.tau)
        self.target_policy.load_state_dict(target_net_state_dict)


        return policy_loss.item()
    
    def get_action(self, state, steps_done=0, eval=False):
        sample = random.random()
        eps_threshold = self.config.eps_end + (self.config.eps_start - self.config.eps_end) * \
            math.exp(-1. * steps_done / self.config.eps_decay)
        if eval or sample > eps_threshold:
            with torch.no_grad():
                # pick action with the larger expected reward.
                state = torch.from_numpy(state).to(self.device, torch.float32)
                return self.policy(state).max(-1)[1].cpu().numpy()
        else:
            return self.env.action_space.sample()
    
    def train_model(self):     
        step_num = 0
        while step_num < self.config.max_steps: 
            t0 = time.time()
            done = False
            rewards = []
            state, _ = self.env.reset()
            while not done:
                with self.ctx:
                    action = self.get_action(state, step_num)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated

                self.memory.push(torch.from_numpy(state), 
                                torch.tensor(action),
                                torch.from_numpy(next_state),
                                reward, 
                                float(not(done)))
                state = next_state

                step_num += 1
                if self.config.warmup_steps >= step_num:
                    continue

                rewards.append(reward)

                policy_loss = self.optimize()

            # dont log or eval during warmpup
            if self.config.warmup_steps >= step_num:
                    continue
            
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            ep_reward = sum(rewards)
            if self.epoch % self.config.log_interval == 0:
                print(f"""epoch {self.epoch} / step {step_num}: policy loss {policy_loss} - episode reward {ep_reward:.4f} - time {dt*1000:.2f}ms""")
            
            if self.epoch > 0 and self.epoch % self.eval_interval == 0:
                self.eval()
                self.eval_model(self.epoch, step_num)
                self.train()
            
            self.epoch += 1
            

    @torch.no_grad()
    def eval_model(self, train_epoch=0, train_step_num=0, save=True):
        print("="*100)
        print("Testing model...".center(100))
        print("="*100)
        
        eval_step_num = 0
        total_rewards = []
        for epoch in range(self.n_eval_epochs):
            t0 = time.time()
            done = False
            rewards = []
            transitions = []
            state, _ = self.env.reset()
            while not done:
                with self.ctx:
                    action = self.get_action(state, eval=True)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated

                rewards.append(reward)
                transitions.append(Transition(torch.from_numpy(state), 
                                torch.tensor(action),
                                torch.from_numpy(next_state),
                                reward, 
                                float(not(done))))
                
                state = next_state
                eval_step_num += 1

            batch = Transition(*zip(*transitions))
            states = torch.stack(batch.state).to(self.device, torch.float32)
            actions = torch.stack(batch.action).to(self.device, torch.long).view(-1, 1)
            next_states = torch.stack(batch.next_state).to(self.device, torch.float32)
            rewards = torch.tensor(batch.reward).to(self.device, torch.float32)
            terminal = torch.tensor(batch.terminal).to(self.device)

            loss = self.compute_loss(states, actions, next_states, rewards, terminal).item()
            
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            rewards = sum(rewards)
            total_rewards.append(rewards)
            print(f"""eval epoch {epoch} / step {eval_step_num}: loss {loss} - episode reward {rewards:.4f} - time {dt*1000:.2f}ms""")

            if save and rewards >= self.best_mean_reward and loss > self.best_loss:
                print("Saving model...")
                self.best_mean_reward = rewards
                self.best_loss = loss
                self.save(train_epoch+1, train_step_num+1)
        
        if self.config.wandb_log:
            wandb.log({
                "iter": train_step_num,
                "rewards": np.mean(total_rewards),
                })
        
        print("="*100)
        print("Finished Testing!".center(100))
        print("="*100)


    def save(self, epoch=None, step_num=None):
        super().save({'model_state_dict': self.state_dict(),
                    'optimizers': {
                        'policy_optimizer': self.policy_optimizer.state_dict()
                    },
                    'config': self.config,
                    'best_mean_reward': self.best_mean_reward,
                    'best_loss': self.best_loss,
                    'epoch': epoch,
                    'step_num': step_num,})
        
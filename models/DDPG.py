"""

Deep Deterministic Policy Gradient (DDPG) implementation.

"""

import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from models.shared.base_model import BaseModel
from models.shared.data import Transition, ReplayMemory
from models.shared.core import DeterministicPolicy, ActionValue, polyak_update


class DDPG(BaseModel):

    def __init__(self, config, env):
        super().__init__(config, env)

        self.policy = DeterministicPolicy(self.n_observations, self.n_actions, hidden_layers=config.policy_hidden_n_layers, 
                                        hidden_sizes=self.config.policy_hidden_sizes, hidden_activation='relu', final_activation='tanh',
                                        frame_stack=self.config.frame_stack, bias=self.config.bias)
        
        self.target_policy = DeterministicPolicy(self.n_observations, self.n_actions, hidden_layers=config.policy_hidden_n_layers, 
                                        hidden_sizes=self.config.policy_hidden_sizes, hidden_activation='relu', final_activation='tanh',
                                        frame_stack=self.config.frame_stack, bias=self.config.bias)
        
        self.q = ActionValue(self.n_observations, self.n_actions, hidden_layers=config.q_hidden_n_layers, hidden_sizes=config.q_hidden_sizes)
        self.target_q = ActionValue(self.n_observations, self.n_actions, hidden_layers=config.q_hidden_n_layers, hidden_sizes=config.q_hidden_sizes)
        
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_q.load_state_dict(self.q.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.policy_lr) 
        self.q_optimizer = optim.Adam(self.q.parameters(), lr=config.q_lr) 

        self.memory = ReplayMemory(self.config.memory_size)

        self.best_policy_loss = float('inf')
        self.action_high = torch.tensor(self.env.action_space.high).to(self.device)
        self.action_low = torch.tensor(self.env.action_space.low).to(self.device)
    
    def get_q_loss(self, states, actions, next_states, rewards, terminal):
        with torch.no_grad():
            target_actions = self.target_policy(next_states)
            target_q_value = self.target_q(next_states, target_actions).view(-1)
            # Compute the expected Q values
            target = rewards + self.config.gamma * target_q_value * terminal

        q_value = self.q(states, actions)
        q_loss = F.mse_loss(q_value.view(-1), target)
        return q_loss
    
    def get_policy_loss(self, states):
        return -self.q(states, self.policy(states)).mean()

    def optimize(self, transitions):
        batch = Transition(*zip(*transitions))
   
        states = torch.stack(batch.state).to(self.device, torch.float32)
        actions = torch.stack(batch.action).to(self.device, torch.long)
        next_states = torch.stack(batch.next_state).to(self.device, torch.float32)
        rewards = torch.tensor(batch.reward).to(self.device, torch.float32)
        terminal = torch.tensor(batch.terminal).to(self.device)

        q_loss = self.get_q_loss(states, actions, next_states, rewards, terminal)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        # gradient clipping so that it doesnt blow up
        torch.nn.utils.clip_grad_value_(self.q.parameters(), 100)
        self.q_optimizer.step()

        policy_loss = self.get_policy_loss(states)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # gradient clipping so that it doesnt blow up
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100)
        self.policy_optimizer.step()

        polyak_update(self.policy, self.target_policy, self.config.tau)
        polyak_update(self.q, self.target_q, self.config.tau)

        return policy_loss.item(), q_loss.item()
    
    @torch.no_grad()
    def get_action(self, state, steps_done=0, eval=False):
        if eval or steps_done > self.config.warmup_steps:
            # pick action with the larger expected reward.
            state = torch.from_numpy(state).to(self.device, torch.float32)
            if eval:
                noise = 0
            else:
                noise = torch.distributions.Normal(0, self.config.gaussian_noise_std * self.action_high).sample()
            action = self.policy(state)
            action = torch.clamp(action + noise, self.action_low, self.action_high)
            return action.cpu().numpy()
        else:
            return self.env.action_space.sample()
    
    def train_model(self):     
        step_num = 0
        while step_num < self.config.max_steps: # epoch starts from 1
            t0 = time.time()
            rewards = []
            terminal = []
            state, _ = self.env.reset()
            epoch_steps = 0
            while epoch_steps < self.config.steps_per_epoch:
                for t in range(self.config.max_ep_len): # limit episode length
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
                    epoch_steps += 1

                    if done:
                        state, _ = self.env.reset()

                    if self.config.warmup_steps >= step_num:
                        continue

                    rewards.append(reward)
                    terminal.append(float(done))
                    
                    if epoch_steps > self.config.update_after and epoch_steps % self.config.update_every == 0:
                        for _ in range(self.config.update_steps):
                            policy_loss, q_loss = self.optimize(self.memory.sample(self.config.batch_size))

            # dont log or eval during warmpup
            if self.config.warmup_steps >= step_num:
                    continue
            
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            terminal = torch.tensor(terminal, device=self.device)
            num_episodes = max(torch.count_nonzero(terminal), 1)
            mean_reward = sum(rewards) / num_episodes
            if self.epoch % self.config.log_interval == 0:
                print(f"""epoch {self.epoch} / step {step_num}: policy loss {policy_loss} - q loss {q_loss} - mean episode reward {mean_reward:.4f} - time {dt*1000:.2f}ms""")

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
            state, _ = self.env.reset()
            rewards = []
            transitions = []
            for t in range(self.config.max_ep_len):
                with self.ctx:
                    action = self.get_action(state, eval=True)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated

                transitions.append(Transition(torch.from_numpy(state), 
                                torch.tensor(action),
                                torch.from_numpy(next_state),
                                reward, 
                                float(not(done))))
 
                rewards.append(reward)
                state = next_state
                eval_step_num += 1
                if done:
                    break

            batch = Transition(*zip(*transitions))
            states = torch.stack(batch.state).to(self.device, torch.float32)
            actions = torch.stack(batch.action).to(self.device, torch.long)
            next_states = torch.stack(batch.next_state).to(self.device, torch.float32)
            rewards = torch.tensor(batch.reward).to(self.device, torch.float32)
            terminal = torch.tensor(batch.terminal).to(self.device)

            q_loss = self.get_q_loss(states, actions, next_states, rewards, terminal)
            policy_loss = self.get_policy_loss(states)
            
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            rewards = sum(rewards)
            total_rewards.append(rewards)
            print(f"""eval epoch {epoch+1} / step {eval_step_num}: policy loss {policy_loss} - q loss {q_loss} - episode reward {rewards:.4f} - time {dt*1000:.2f}ms""")

            if save and rewards >= self.best_mean_reward and policy_loss < self.best_policy_loss:
                print("Saving model...")
                self.best_mean_reward = rewards
                self.best_policy_loss = policy_loss
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
                        'policy_optimizer': self.policy_optimizer.state_dict(),
                        'q_optimizer': self.q_optimizer.state_dict(),
                    },
                    'config': self.config,
                    'best_mean_reward': self.best_mean_reward,
                    'best_policy_loss': self.best_policy_loss,
                    'epoch': epoch,
                    'step_num': step_num,})
        
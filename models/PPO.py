"""

Proximal Policy Optimization (PPO) implementation.

PPO-Clip variant of PPO.

"""

import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from models.shared.base_model import BaseModel
from models.shared.data import Transition
from models.shared.core import StochasticPolicy, Value
from models.shared.math import estimate_advantage_with_value_fn, get_action_log_prob


class PPO(BaseModel):

    def __init__(self, config, env):
        super().__init__(config, env)

        self.policy = StochasticPolicy(self.n_observations, self.n_actions, hidden_layers=config.policy_hidden_n_layers, 
                                        hidden_sizes=self.config.policy_hidden_sizes, hidden_activation='tanh', action_space=env.action_space, 
                                        frame_stack=self.config.frame_stack, bias=self.config.bias, log_sig_min=self.config.log_sig_min, 
                                        log_sig_max=self.config.log_sig_max, epsilon=self.config.epsilon, discrete=self.discrete_action_space)

        self.value_fn = Value(self.n_observations, hidden_layers=config.value_hidden_n_layers, hidden_sizes=self.config.value_hidden_sizes, hidden_activation='tanh')
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=config.policy_lr) 
        self.value_optimizer = optim.Adam(self.value_fn.parameters(), lr=config.value_lr) 
        
        self.best_value_loss = -float('inf')
    
    def compute_value_loss(self, states, targets):
        values = self.value_fn(states)
        return F.mse_loss(values.flatten(), targets.flatten())
    
    def update_value_fn(self, states, targets):
        loss = self.compute_value_loss(states, targets)
        self.value_optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.value_optimizer)
        # Updates the scale for next iteration.
        self.scaler.update()
        return loss.item()  
    
    def train_model(self):
        step_num = 0
        while step_num < self.config.max_steps:
            t0 = time.time()
            transitions = self.sample_batch_from_env()
            batch = Transition(*zip(*transitions))
            
            states = torch.stack(batch.state).to(self.device, torch.float32)
            actions = torch.stack(batch.action).to(self.device, torch.float32)
            rewards = torch.tensor(batch.reward).to(self.device, torch.float32)
            terminal = torch.tensor(batch.terminal).to(self.device)
            
            with torch.no_grad():
                advantages, returns = estimate_advantage_with_value_fn(states, rewards, terminal, self.value_fn, self.config.gamma)
                # Normalize advantages so that gradients arent too large (can improve convergence but may not matter much)
                advantages = (advantages - advantages.mean()) / advantages.std()
       
                log_p_old = get_action_log_prob(self.policy, states, actions, ctx=self.ctx)

            for _ in range(self.config.update_steps):
                value_loss = self.update_value_fn(states, returns)

                new_log_p = get_action_log_prob(self.policy, states, actions, ctx=self.ctx)
                ratio = torch.exp(new_log_p - log_p_old)
                policy_loss_unclipped = -advantages * ratio
                policy_loss_clipped = -advantages * torch.clamp(ratio, 1-self.config.clip_ratio, 1+self.config.clip_ratio)
                
                policy_entropy = (new_log_p.exp() * new_log_p).mean()
                policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean() + policy_entropy * self.config.lambda_entropy

                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()
                self.scaler.step(self.policy_optimizer)
                # Updates the scale for next iteration.
                self.scaler.update()

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            step_num += len(actions)
            num_episodes = max(terminal.shape[0] - torch.count_nonzero(terminal), 1)
            mean_reward = sum(rewards) / num_episodes
            print(f"""epoch {self.epoch} / step {step_num}: policy loss {policy_loss} - value loss {value_loss:.4f} - mean reward {mean_reward:.4f} - time {dt*1000:.2f}ms""")

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
        
        step_num = 0
        for epoch in range(self.n_eval_epochs):
            t0 = time.time()
            transitions = self.sample_batch_from_env()
            batch = Transition(*zip(*transitions))
            
            states = torch.stack(batch.state).to(self.device, torch.float32)
            actions = torch.stack(batch.action).to(self.device, torch.float32)
            rewards = torch.tensor(batch.reward).to(self.device, torch.float32)
            terminal = torch.tensor(batch.terminal).to(self.device)

            with torch.no_grad():
                advantages, returns = estimate_advantage_with_value_fn(states, rewards, terminal, self.value_fn, self.config.gamma)
                # Normalize advantages so that gradients arent too large (can improve convergence but may not matter much)
                advantages = (advantages - advantages.mean()) / advantages.std()
       
                log_p_old = get_action_log_prob(self.policy, states, actions, ctx=self.ctx)

            value_loss = self.compute_value_loss(states, returns)

            new_log_p = get_action_log_prob(self.policy, states, actions, ctx=self.ctx)
            ratio = torch.exp(new_log_p - log_p_old)
            policy_loss_unclipped = -advantages * ratio
            policy_loss_clipped = -advantages * torch.clamp(ratio, 1-self.config.clip_ratio, 1+self.config.clip_ratio)
            
            policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            step_num += len(actions)
            num_episodes = max(terminal.shape[0] - torch.count_nonzero(terminal), 1)
            mean_reward = sum(rewards) / num_episodes
            print(f"""eval epoch {epoch} / step {step_num}: policy loss {policy_loss:.8f} - value loss {value_loss:.4f} - mean reward {mean_reward:.4f} - time {dt*1000:.2f}ms""")

            if save and mean_reward >= self.best_mean_reward and value_loss > self.best_value_loss:
                print("Saving model...")
                self.best_mean_reward = mean_reward
                self.best_policy_loss = policy_loss
                self.best_value_loss = value_loss
                self.save(train_epoch+1, train_step_num+1)

            if self.config.wandb_log:
                wandb.log({
                    "epoch": epoch,
                    "iter": step_num,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "rewards": rewards,
                    "time_per_epoch": round(dt*1000, 2),
                    "advantages": advantages,
                    })
        print("="*100)
        print("Finished Testing!".center(100))
        print("="*100)


    def save(self, epoch=None, step_num=None):
        super().save({'model_state_dict': self.state_dict(),
                    'optimizers': {
                        'policy_optimizer': self.policy_optimizer.state_dict(),
                        'value_optimizer': self.value_optimizer.state_dict()
                    },
                    'config': self.config,
                    'best_mean_reward': self.best_mean_reward,
                    'best_policy_loss': self.best_policy_loss,
                    'best_value_loss': self.best_value_loss,
                    'epoch': epoch,
                    'step_num': step_num,})
        
"""

Vanilla Policy Gradient (VPG)

"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from models.shared.base_model import BaseModel
from models.shared.utils import init_weights
from models.shared.core import StochasticPolicy, DeterministicPolicy, Value


class VPG(BaseModel):

    def __init__(self, config, env):
        super().__init__(config, env)

        self.n_epochs = config.n_epochs

        if self.discrete_action_space:
            self.policy = DeterministicPolicy(self.n_observations, self.n_actions, hidden_layers=config.policy_hidden_n_layers, 
                                              hidden_sizes=self.config.policy_hidden_sizes, hidden_activation='relu', 
                                              final_activation='softmax', frame_stack=self.config.frame_stack, bias=self.config.bias)
        else:
            self.policy = StochasticPolicy(self.n_observations, self.n_actions, hidden_layers=config.policy_hidden_n_layers, 
                                           hidden_sizes=self.config.policy_hidden_sizes, hidden_activation='tanh', action_space=env.action_space, 
                                           frame_stack=self.config.frame_stack, bias=self.config.bias, log_sig_min=self.config.log_sig_min, 
                                           log_sig_max=self.config.log_sig_max, epsilon=self.config.epsilon)

        self.value_fn = Value(self.n_observations, hidden_layers=config.value_hidden_n_layers, hidden_sizes=self.config.value_hidden_sizes, hidden_activation='tanh')
        
        self.value_optimizer = optim.Adam(self.value_fn.parameters(), lr=config.value_lr) 
    

    """ def sample_policy(self, state):
        logits = self.policy(state)
        return Categorical(logits=logits)
    
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return self.sample_policy(state).sample().item() """

    def compute_loss(self, state, action, rewards_to_go):
        log_prob = self.sample_policy(state).log_prob(action)
        return -(log_prob * rewards_to_go).mean() # negative to perform gradient ascent

    def reward_to_go(self, rews):
        """ Calculate the reward-to-go which only includes present and future rewards. """
        n = len(rews)
        rtgs = torch.zeros_like(torch.tensor(rews), device=self.device, dtype=torch.float32)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs
    
    def sample_batch_from_env(self):
        """ Sample a batch of trajectories from the environment. """

        batch_states, batch_actions, batch_rewards_to_go, batch_returns, batch_lengths = [], [], [], [], []
        ep_rews = []
        state, _ = self.env.reset()
        while True:
            action = self.get_action(state)
            next_state, reward, terminated, _, _ = self.env.step(action) 
            done = terminated

            batch_states.append(state)
            batch_actions.append(action)
            ep_rews.append(reward)

            # Move to the next state
            state = next_state
            
            if done:
                # Episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_returns.append(ep_ret)
                batch_lengths.append(ep_len)

                batch_rewards_to_go += self.reward_to_go(ep_rews).tolist()

                state, _ = self.env.reset()
                ep_rews = []

                # end experience loop if we have enough of it
                if len(batch_states) > self.batch_size:
                    break
        return np.array(batch_states), np.array(batch_actions), np.array(batch_rewards_to_go), batch_returns, batch_lengths
            
            
    def train_model(self):
        print(self.training)
        step_num = 0
        for epoch in range(self.num_epochs):
            t0 = time.time()

            batch_states, batch_actions, batch_rewards_to_go, batch_returns, batch_lengths = self.sample_batch_from_env()

            loss = self.compute_loss(torch.as_tensor(batch_states, device=self.device), 
                                     torch.as_tensor(batch_actions, device=self.device), 
                                     torch.as_tensor(batch_rewards_to_go, device=self.device)
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            step_num += sum(batch_lengths)
            mean_reward = np.mean(batch_returns)
            print(f"""epoch {epoch} / step {step_num}: policy loss {loss:.4f} - mean reward {mean_reward:.4f} - time {dt*1000:.2f}ms""")

            if epoch % self.eval_interval == 0:
                self.eval_model()

    @torch.no_grad()
    def eval_model(self, save=True):
        print("="*100)
        print("Testing model...".center(100))
        print("="*100)
        step_num = 0
        for epoch in range(self.eval_episodes):
            t0 = time.time()

            batch_states, batch_actions, batch_rewards_to_go, batch_returns, batch_lengths = self.sample_batch_from_env()
            
            loss = self.compute_loss(torch.tensor(batch_states, device=self.device), 
                                     torch.tensor(batch_actions, device=self.device), 
                                     torch.tensor(batch_rewards_to_go, device=self.device)
            )
        
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            step_num += sum(batch_lengths)
            mean_reward = np.mean(batch_returns)
            print(f"""epoch {epoch} / step {step_num}: policy loss {loss:.4f} - mean reward {mean_reward:.4f} - time {dt*1000:.2f}ms""")

            if save and mean_reward > self.best_reward:
                print("Saving model...")
                self.best_reward = mean_reward
                self.save()
        print("="*100)
        print("Finished Testing!".center(100))
        print("="*100)


    def save(self):
        path = os.path.join(self.config.out_dir, 'model.pt')
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config,
                    'best_reward': self.best_reward}, path)

    def load(self, checkpoint):
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.best_reward = checkpoint['best_reward']

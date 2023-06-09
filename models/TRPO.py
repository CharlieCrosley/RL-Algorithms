""" 

Trust Region Policy Optimization (TRPO) implementation.

Helpful resources:
- https://spinningup.openai.com/en/latest/algorithms/trpo.html
- https://medium.com/@vladogim97/trpo-minimal-pytorch-implementation-859e46c4232e
- https://github.com/ikostrikov/pytorch-trpo/tree/master

"""

import os
import time
import torch
import torch.optim as optim
from torch.functional import F
import wandb
from models.shared.base_model import BaseModel
from models.shared.utils import apply_parameter_update, apply_parameter_update, get_flat_params_from
from models.shared.data import Transition
from models.shared.core import Value, StochasticPolicy, flat_grad
from models.shared.math import conjugate_gradient, compute_surrogate_loss_and_kl, estimate_advantage_with_value_fn


class TRPO(BaseModel):

    def __init__(self, config, env):
        super().__init__(config, env)

        
        self.policy = StochasticPolicy(self.n_observations, self.n_actions, hidden_layers=config.policy_hidden_n_layers, 
                                        hidden_sizes=self.config.policy_hidden_sizes, hidden_activation='tanh', action_space=env.action_space, 
                                        frame_stack=self.config.frame_stack, bias=self.config.bias, log_sig_min=self.config.log_sig_min, 
                                        log_sig_max=self.config.log_sig_max, epsilon=self.config.epsilon, discrete=self.discrete_action_space)

        self.value_fn = Value(self.n_observations, hidden_layers=config.value_hidden_n_layers, hidden_sizes=self.config.value_hidden_sizes, hidden_activation='tanh')
        self.value_optimizer = optim.Adam(self.value_fn.parameters(), lr=config.value_lr) 

        self.best_value_loss = -float('inf')

    def compute_value_loss(self, states, targets):
        values = self.value_fn(states)
        return F.mse_loss(values, targets)
    
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

            advantages, returns = estimate_advantage_with_value_fn(states, rewards, terminal, self.value_fn, discount=self.config.gamma)
            # Normalize advantages to help stabilize learning (can improve convergence but may not matter much)
            advantages = (advantages - advantages.mean()) / advantages.std()

            policy_loss, fixed_log_probs, kl = compute_surrogate_loss_and_kl(self.policy, states, actions, advantages, ctx=self.ctx)

            g = flat_grad(policy_loss, self.policy.parameters(), retain_graph=True).data # We will use the graph several times
            d_kl = flat_grad(kl, self.policy.parameters(), create_graph=True)  # Create graph, because we will call backward() on it (for HVP)
        
            def hessian_vector_product(v):
                """ Compute the Hessian-vector product """
                return flat_grad(d_kl @ v, self.policy.parameters(), retain_graph=True) + v * self.config.damping_coeff
            
            step_dir = conjugate_gradient(hessian_vector_product, g, max_iterations=self.config.cg_iters)
            
            max_length = torch.sqrt(2 * self.config.max_kl_divergence / (step_dir @ hessian_vector_product(step_dir)))
            max_step = max_length * step_dir
            
            old_loss = policy_loss

            @torch.no_grad()
            def backtracking_line_search():
                prev_params = get_flat_params_from(self.policy)
                for i in range(self.config.backtrack_iters):
                    step = (self.config.backtrack_coeff ** i) * max_step
                    param_new = prev_params + step
                    apply_parameter_update(self.policy.parameters(), param_new)

                    new_loss, _, kl_new = compute_surrogate_loss_and_kl(self.policy, states, actions, advantages, old_log_probs=fixed_log_probs, ctx=self.ctx)
                    loss_improvement = new_loss - old_loss
                    if loss_improvement > 0 and kl_new <= self.config.max_kl_divergence:
                        return True
                return False
            
            backtracking_line_search()

            for _ in range(self.config.update_steps):
                value_fn_loss = self.update_value_fn(states, returns)
            
            # timing and logging
            t1 = time.time()
            dt = t1 - t0

            step_num += len(actions)
            # terminal values of zero indicate end of episode
            num_episodes = max(terminal.shape[0] - torch.count_nonzero(terminal), 1)
            mean_reward = sum(rewards) / num_episodes
            print(f"""epoch {self.epoch} / step {step_num}: policy loss {policy_loss} - value loss {value_fn_loss:.4f} - mean reward {mean_reward:.4f} - time {dt*1000:.2f}ms""")

            if self.config.wandb_log:
                wandb.log({
                    "epoch": self.epoch,
                    "iter": step_num,
                    "policy_loss": policy_loss,
                    "value_loss": value_fn_loss,
                    "mean_reward": mean_reward,
                    "time_per_epoch": round(dt*1000, 2),
                    "kl_divergence": kl,
                    "advantages": advantages,
                    })

            if self.epoch > 0 and self.epoch % self.eval_interval == 0:
                self.eval_model(self.epoch, step_num)

            self.epoch += 1

    @torch.no_grad()
    def eval_model(self, train_epoch=0, train_step_num=0, save=True):
        print("="*100)
        print("Testing model...".center(100))
        print("="*100)
        self.eval()
        step_num = 0
        for epoch in range(self.n_eval_epochs):
            t0 = time.time()

            trajectories = self.sample_batch_from_env()
            
            batch = Transition(*zip(*trajectories))
            
            states = torch.stack(batch.state).to(self.device, torch.float32)
            actions = torch.stack(batch.action).to(self.device, torch.float32)
            rewards = torch.tensor(batch.reward).to(self.device, torch.float32)
            terminal = torch.tensor(batch.terminal).to(self.device)

            advantages, returns = estimate_advantage_with_value_fn(states, rewards, terminal, self.value_fn, discount=self.config.gamma)
            # Normalize advantages so that gradients arent too large (can improve convergence but may not matter much)
            advantages = (advantages - advantages.mean()) / advantages.std()

            value_fn_loss = self.compute_value_loss(states, returns)

            policy_loss, _, _ = compute_surrogate_loss_and_kl(self.policy, states, actions, advantages, ctx=self.ctx)
        
            # timing and logging
            t1 = time.time()
            dt = t1 - t0

            step_num += len(actions)
            # terminal values of zero indicate end of episode
            num_episodes = terminal.shape[0] - torch.count_nonzero(terminal)
            num_episodes = num_episodes if num_episodes > 0 else 1
            mean_reward = sum(rewards) / num_episodes
            print(f"""eval epoch {epoch+1} / step {step_num}: policy loss {policy_loss} - value loss {value_fn_loss:.4f} - mean reward {mean_reward:.4f} - time {dt*1000:.2f}ms""")

            if save and mean_reward >= self.best_mean_reward and value_fn_loss > self.best_value_loss:
                print("Saving model...")
                self.best_mean_reward = mean_reward
                self.best_policy_loss = policy_loss
                self.best_value_loss = value_fn_loss
                self.save(train_epoch+1, train_step_num+1)
        self.train()
        print("="*100)
        print("Finished Testing!".center(100))
        print("="*100)


    def save(self, epoch=None, step_num=None):
        path = os.path.join(self.config.out_dir, 'model.tar')
        torch.save({'model_state_dict': self.state_dict(),
                    'value_optimizer_state_dict': self.value_optimizer.state_dict(),
                    'config': self.config,
                    'best_mean_reward': self.best_mean_reward,
                    'best_policy_loss': self.best_policy_loss,
                    'best_value_loss': self.best_value_loss,
                    'epoch': epoch,
                    'step_num': step_num,
                    }, path)

    def load(self, checkpoint):
        self.load_state_dict(checkpoint['model_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.config = checkpoint['config']
        self.best_mean_reward = checkpoint['best_mean_reward']
        self.best_policy_loss = checkpoint['best_policy_loss']
        self.best_value_loss = checkpoint['best_value_loss']
        self.epoch = checkpoint['epoch']
        self.step_num = checkpoint['step_num']
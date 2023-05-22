import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size, bias=False):
        super().__init__()


class ActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, bias=False):
        super().__init__()


class StochasticPolicy(nn.Module):
    def __init__(self, config, n_observations, n_actions, action_space):
        super().__init__()

        self.config = config
        self.linear1 = nn.Linear(config.n_frame_stack * n_observations, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.mean_linear = nn.Linear(config.hidden_size, n_actions)
        self.log_std_linear = nn.Linear(config.hidden_size, n_actions)
        
        # action rescaling
        if action_space is None:
            self.register_buffer('action_scale', torch.tensor(1.))
            self.register_buffer('action_bias', torch.tensor(0.))
        else:
            self.register_buffer('action_scale', torch.tensor(
                (action_space.high - action_space.low) / 2.))
            self.register_buffer('action_bias', torch.tensor(
                (action_space.high + action_space.low) / 2.))
            
        #log_std = -0.5 * torch.ones(n_actions, dtype=torch.float32)
        #self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        #self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        #print(self.mean_linear.weight.data[:1, :10])
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.config.log_sig_min, max=self.config.log_sig_max)
        return mean, log_std
    
    def sample(self, state):
        #print(state)
        mean, log_std = self.forward(state)
        std = log_std.exp() # We use log_std because they always have a positive value
        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias


        """ log_prob = normal.log_prob(x_t).sum(axis=-1)
        log_prob -= (2*(np.log(2) - x_t - F.softplus(-2*x_t))).sum(axis=-1) """


        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.config.epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
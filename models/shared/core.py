import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models.shared.utils import create_layers


class Value(nn.Module):
    
    def __init__(self, config, state_dim, hidden_layers=1, hidden_sizes=(64,32), hidden_activation='relu', final_activation=None):
        super().__init__()

        self.layers = create_layers(
            config, 
            state_dim, 
            1, 
            hidden_layers=hidden_layers,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation, 
            final_activation=final_activation
        )
    
    def forward(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)
        return x

class ActionValue(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, bias=False):
        super().__init__()

class DeterministicPolicy(nn.Module):
    def __init__(self, config, n_observations, n_actions, hidden_layers=1, hidden_sizes=(64,32), hidden_activation='relu', final_activation=None):
        super().__init__()
        assert final_activation in ['tanh', 'relu', 'sigmoid', 'softmax', None]
        assert hidden_activation in ['tanh', 'relu', 'sigmoid', 'softmax', None]
        self.layers = create_layers(
            config, 
            n_observations, 
            n_actions, 
            hidden_layers=hidden_layers,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation, 
            final_activation=final_activation
        )

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)
        return x
    
class StochasticPolicy(nn.Module):
    def __init__(self, config, n_observations, n_actions, hidden_layers=1, hidden_sizes=(256,256), hidden_activation='relu', action_space=None):
        super().__init__()

        self.config = config

        self.layers = create_layers(
                    config, 
                    n_observations, 
                    hidden_sizes[-1], 
                    hidden_layers=hidden_layers,
                    hidden_sizes=hidden_sizes,
                    hidden_activation=hidden_activation
                )
        self.mean_linear = nn.Linear(hidden_sizes[-1], n_actions)
        self.log_std_linear = nn.Linear(hidden_sizes[-1], n_actions)
        
        # action rescaling
        if action_space is None:
            self.register_buffer('action_scale', torch.tensor(1.))
            self.register_buffer('action_bias', torch.tensor(0.))
        else:
            self.register_buffer('action_scale', torch.tensor(
                (action_space.high - action_space.low) / 2.))
            self.register_buffer('action_bias', torch.tensor(
                (action_space.high + action_space.low) / 2.))
            

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)
        
        mean = self.mean_linear(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.config.log_sig_min, max=self.config.log_sig_max)

        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp() # We use log_std because they always have a positive value
        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.config.epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, log_std, normal
    
def flat_grad(x, parameters, retain_graph=False, create_graph=False):
    """ Compute gradient of x w.r.t parameters and flatten it """
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(x, parameters, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from models.shared.utils import create_layers


class Value(nn.Module):
    
    def __init__(self, n_observations, hidden_layers=1, hidden_sizes=(64,32), hidden_activation='relu', final_activation=None, frame_stack=1, bias=True):
        super().__init__()

        self.layers = create_layers( 
            n_observations, 
            1, 
            hidden_layers=hidden_layers,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation, 
            final_activation=final_activation,
            frame_stack=frame_stack,
            bias=bias
        )
        
    
    def forward(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)
        return x

class QNetwork(nn.Module):
    
    def __init__(self, n_observations, n_actions, hidden_layers=1, hidden_sizes=(64,32), hidden_activation='relu', final_activation=None, frame_stack=1, bias=True, discrete=False):
        super().__init__()

        if discrete:
            action_size = 1
        else:
            action_size = n_actions

        self.layers = create_layers( 
            n_observations + action_size,
            1, 
            hidden_layers=hidden_layers,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation, 
            final_activation=final_activation,
            frame_stack=frame_stack,
            bias=bias
        )
        
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x
    
class DoubleQNetwork(nn.Module):
    
    def __init__(self, n_observations, n_actions, hidden_layers=1, hidden_sizes=(64,32), hidden_activation='relu', final_activation=None, frame_stack=1, bias=True, discrete=False):
        super().__init__()

        if discrete:
            action_size = 1
        else:
            action_size = n_actions

        self.q1 = create_layers( 
            n_observations + action_size,
            1, 
            hidden_layers=hidden_layers,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation, 
            final_activation=final_activation,
            frame_stack=frame_stack,
            bias=bias
        )
        self.q2 = create_layers( 
            n_observations + action_size,
            1, 
            hidden_layers=hidden_layers,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation, 
            final_activation=final_activation,
            frame_stack=frame_stack,
            bias=bias
        )
        
    
    def forward(self, state, action):
        q1 = torch.cat([state, action], dim=1)
        q2 = q1.clone()
        for layer in self.q1:
            q1 = layer(q1)
        for layer in self.q2:
            q2 = layer(q2)
        return q1, q2
    
    def Q1(self, state, action):
        q1 = torch.cat([state, action], dim=1)
        for layer in self.q1:
            q1 = layer(q1)
        return q1


class DeterministicPolicy(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_layers=1, hidden_sizes=(64,32), hidden_activation='relu', final_activation=None, frame_stack=1, bias=True):
        super().__init__()
        assert final_activation in ['tanh', 'relu', 'sigmoid', 'softmax', None]
        assert hidden_activation in ['tanh', 'relu', 'sigmoid', 'softmax', None]
        self.layers = create_layers(
            n_observations, 
            n_actions, 
            hidden_layers=hidden_layers,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation, 
            final_activation=final_activation,
            frame_stack=frame_stack,
            bias=bias
        )

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)
        return x
    
class StochasticPolicy(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_layers=1, hidden_sizes=(256,256), hidden_activation='relu', 
                 action_space=None, frame_stack=1, bias=True, log_sig_min=-20, log_sig_max=2, epsilon=1e-8, discrete=False):
        super().__init__()

        self.layers = create_layers(
                    n_observations, 
                    hidden_sizes[-1], 
                    hidden_layers=hidden_layers,
                    hidden_sizes=hidden_sizes,
                    hidden_activation=hidden_activation,
                    frame_stack=frame_stack,
                    bias=bias
                )
        self.mean_linear = nn.Linear(hidden_sizes[-1], n_actions)
        self.log_std_linear = nn.Linear(hidden_sizes[-1], n_actions)

        self.log_sig_min = log_sig_min
        self.log_sig_max = log_sig_max
        self.epsilon = epsilon
        
        # action rescaling
        if action_space is None or discrete:
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
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)

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
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, log_std
    
def flat_grad(x, parameters, retain_graph=False, create_graph=False):
    """ Compute gradient of x w.r.t parameters and flatten it """
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(x, parameters, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g

def polyak_update(policy, target, tau=0.005):
    """ Polyak update of the parameters """
    
    state_dict = policy.state_dict()
    target_state_dict = target.state_dict()
    for key in state_dict:
        target_state_dict[key] = state_dict[key]*tau + target_state_dict[key]*(1-tau)
    target.load_state_dict(target_state_dict)
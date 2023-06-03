from contextlib import nullcontext
import os
import numpy as np
from torch.distributions import Categorical
import torch
from torch.cuda.amp import GradScaler
from abc import ABC, abstractmethod

from models.shared.core import DeterministicPolicy
from models.shared.data import Transition
from operator import attrgetter

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}

# Abstract Base Class for all models
class BaseModel(torch.nn.Module, ABC):

    def __init__(self, config, env):
        super(BaseModel, self).__init__()
        assert config.policy_hidden_n_layers+1 == len(config.policy_hidden_sizes)
        if hasattr(config, 'value_hidden_n_layers'):
            assert config.value_hidden_n_layers+1 == len(config.value_hidden_sizes)
        if hasattr(config, 'q_hidden_n_layers'):
            assert config.q_hidden_n_layers+1 == len(config.q_hidden_sizes)
        
        self.env = env
        self.device = config.device
        self.config = config
        self.batch_size = config.batch_size
        self.n_eval_epochs = config.n_eval_epochs
        self.eval_interval = config.eval_interval
        self.epoch = 0
        self.n_epochs = config.n_epochs

        self.best_mean_reward = -float('inf')
        
        self.n_observations = env.observation_space.shape[0]
        
        if len(env.action_space.shape) == 0:
            # Discrete action space
            self.n_actions = env.action_space.n
            self.discrete_action_space = True
        else:
            # Continuous action space
            self.n_actions = env.action_space.shape[0]
            self.discrete_action_space = False

        # Used for mixed precision training, which can improve performance especially for linear layers
        self.dtype = ptdtype[self.config.dtype]
        self.ctx = nullcontext() if self.device == 'cpu' else torch.amp.autocast(device_type=self.device, dtype=self.dtype)
        self.scaler = GradScaler(enabled=(self.config.dtype == 'float16'))

        # logging
        if config.wandb_log:
            import wandb
            wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=config)

    @torch.no_grad()
    def get_action(self, state):
        """Get action from the model."""
        
        state = torch.from_numpy(state).to(device=self.device, dtype=self.dtype)
        if isinstance(self.policy, DeterministicPolicy):
            if self.training:
                dist = Categorical(self.policy(state))  # Create a distribution from logits for actions
                return dist.sample().item()
            else:
                return self.policy(state).argmax().item()
        else:
            action, _, mean, _ = self.policy.sample(state)
            if not self.training:
                action = mean # take the mean action during evaluation
            
            #print(action)
            return action.cpu().numpy()


    def sample_batch_from_env(self, enable_truncation=False):
        """ Sample a batch of transitions from the environment. """

        transitions = []
        state, _ = self.env.reset()
        # sample a batch of trajectories
        for t in range(self.config.steps_per_epoch):
            # Runs the forward pass with autocasting.
            with self.ctx:
                action = self.get_action(state)
                if self.discrete_action_space:       
                    next_state, reward, terminated, truncated, _ = self.env.step(np.argmax(action)) 
                else:
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or (enable_truncation and truncated)

            transitions.append(Transition(torch.from_numpy(state), 
                                       torch.tensor(action),
                                       torch.from_numpy(next_state),
                                       reward, 
                                       float(not(done))))
            if done: 
                state, _ = self.env.reset()
            state = next_state # Move to the next state
        return transitions

    @abstractmethod
    def train_model(self):
        """Train the model."""
        pass

    @abstractmethod
    def eval_model(self, save=True):
        """Evaluate the model."""
        pass
    
    def save(self, save_dict):
        """ Save model parameters """

        for wrapper in self.env.spec.additional_wrappers: # NormalizeObservation wrapper keeps running mean and variance, so we need to save them
            if wrapper.name == 'NormalizeObservation':
                save_dict['env_args'] = {'NormalizeObservation': {}}
                save_dict['env_args']['NormalizeObservation']['env.obs_rms.mean'] = self.env.obs_rms.mean
                save_dict['env_args']['NormalizeObservation']['env.obs_rms.var'] = self.env.obs_rms.var
                save_dict['env_args']['NormalizeObservation']['env.obs_rms.count'] = self.env.obs_rms.count
                save_dict['env_args']['NormalizeObservation']['env.epsilon'] = self.env.epsilon
                break

        path = os.path.join(self.config.out_dir, 'model.tar')
        torch.save(save_dict, path)

    def load(self, checkpoint):
        """ Load model parameters """

        for key, value in checkpoint.items():
            if key == 'optimizers':
                for optim, state_dict in checkpoint['optimizers'].items():
                    self.__dict__[optim].load_state_dict(state_dict)
            elif key == 'model_state_dict':
                self.load_state_dict(value)
            elif key == 'env_args':
                for wrapper, params in checkpoint['env_args'].items(): # kinda gross but it works as long as the dict keys have correct names
                    for param, value in params.items():
                        obj = self
                        var_path = param.split('.')
                        for var in var_path[:-1]:
                            obj = obj.__dict__[var]
                        obj.__dict__[var_path[-1]] = value
            else:
                self.__dict__[key] = value

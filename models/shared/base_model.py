from contextlib import nullcontext
import torch
from torch.cuda.amp import GradScaler
from abc import ABC, abstractmethod

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}

# Abstract Base Class for all models
class BaseModel(torch.nn.Module, ABC):

    def __init__(self, config, env):
        super(BaseModel, self).__init__()
        assert config.policy_hidden_n_layers+1 == len(config.policy_hidden_sizes)
        assert config.value_hidden_n_layers+1 == len(config.value_hidden_sizes)
        
        self.env = env
        self.device = config.device
        self.config = config
        self.batch_size = config.batch_size
        self.n_eval_epochs = config.n_eval_epochs
        self.eval_interval = config.eval_interval
        self.epoch = 0
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

    @abstractmethod
    def get_action(self, state):
        """Get action from the model."""
        pass

    @abstractmethod
    def sample_batch_from_env(self):
        """Sample a batch of transitions from the environment."""
        pass

    @abstractmethod
    def train_model(self):
        """Train the model."""
        pass

    @abstractmethod
    def eval_model(self, save=True):
        """Evaluate the model."""
        pass
    
    @abstractmethod
    def save(self):
        """ Save model parameters """
        pass

    @abstractmethod
    def load(self, checkpoint):
        """ Load model parameters """
        pass
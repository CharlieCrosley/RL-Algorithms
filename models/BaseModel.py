import torch
from abc import ABC, abstractmethod

# Abstract Base Class for all models
class BaseModel(torch.nn.Module, ABC):

    def __init__(self, config, env):
        super(BaseModel, self).__init__()

        self.env = env
        self.device = config.device
        self.config = config
        self.batch_size = config.batch_size
        self.eval_episodes = config.eval_episodes
        self.eval_interval = config.eval_interval
        
        self.n_observations = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.best_reward = -float('inf')

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
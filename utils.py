from collections import deque, namedtuple
from random import sample
import torch

def get_model_obj(model_name):
    """ Returns the model object given the model name """
    from models import VPG
    return {'vpg':VPG.VPG}[model_name]

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))

class ReplayMemory(object):

    def __init__(self, size) -> None:
        self.buffer = deque([], maxlen=size)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    

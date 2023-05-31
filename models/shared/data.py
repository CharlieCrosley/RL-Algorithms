from collections import deque, namedtuple
from random import sample


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))

Trajectory = namedtuple('Trajectory',
                        ['states', 'actions', 'rewards', 'next_states', 'terminal'])

class ReplayMemory(object):

    def __init__(self, size) -> None:
        self.buffer = deque([], maxlen=size)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import deque, namedtuple
import random

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state'))


class DQN(nn.Module):

    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_observations,128),
            nn.Linear(128, 128),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.model(x)
    


class ReplayMemory(object):

    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

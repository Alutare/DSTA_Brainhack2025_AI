import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, output_dim))

    def forward(self, x):
        x = self.feature(x)
        value = self.value(x)
        adv = self.advantage(x)
        return value + adv - adv.mean(dim=1, keepdim=True)

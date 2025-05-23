import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        
        # Replace BatchNorm with LayerNorm which works with any batch size
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),  # LayerNorm instead of BatchNorm
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),  # LayerNorm instead of BatchNorm
            nn.ReLU()
        )
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream with proper initialization
        self.advantage = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Initialize weights for better performance
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.feature(x)
        value = self.value(x)
        adv = self.advantage(x)
        # Dueling architecture
        return value + adv - adv.mean(1, keepdim=True)
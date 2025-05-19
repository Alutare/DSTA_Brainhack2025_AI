import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # Ensure x is 2D for attention
        if x.dim() == 1:
            x = x.unsqueeze(0)
        weights = self.attention(x)
        return x * weights
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        
        # Attention mechanism for focusing on important state features
        self.attention = AttentionLayer(state_dim)
        
        # First feature extraction layer
        self.feature1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),  # LayerNorm for better stability
            nn.ReLU(),
            nn.Dropout(0.1)  # Add dropout for regularization
        )
        
        # Second feature extraction layer with residual connection
        self.feature2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
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
        # Apply attention mechanism
        x = self.attention(x)
        
        # Feature extraction with residual connection
        f1 = self.feature1(x)
        f2 = self.feature2(f1)
        features = f1 + f2  # Residual connection
        
        # Value and advantage streams
        value = self.value(features)
        adv = self.advantage(features)
        
        # Dueling architecture combination
        return value + adv - adv.mean(1, keepdim=True)

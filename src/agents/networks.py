"""
Neural network architectures for the RL agents.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Deep Q-Network for traffic light control.

    A simple 3-layer MLP that maps state observations to Q-values
    for each possible action. Uses ReLU activations and optional
    layer normalization for training stability.

    Architecture:
        Input (state_dim) → 256 → 256 → 128 → Output (action_dim)
    """

    def __init__(self, state_dim=14, action_dim=3, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )

        # Initialize weights (Xavier)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: State tensor of shape (batch, state_dim)

        Returns:
            Q-values tensor of shape (batch, action_dim)
        """
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture (optional upgrade).

    Separates state value V(s) from action advantage A(s,a):
        Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    This helps the network learn which states are valuable
    independently of the action taken.
    """

    def __init__(self, state_dim=14, action_dim=3, hidden_size=256):
        super().__init__()

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        features = self.features(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

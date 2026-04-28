"""
Deep Q-Network (DQN) Agent for traffic light optimization.

Features:
- Experience replay buffer for sample efficiency
- Target network with soft updates for stability
- Epsilon-greedy exploration with decay
- CUDA acceleration for RTX 4060
- Model checkpointing
"""

import random
import math
import os
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import QNetwork

# Named tuple for replay buffer entries
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with target network and experience replay.

    Hyperparameters are tuned for the traffic control problem:
    - Small network (state_dim=14, action_dim=3)
    - Moderate replay buffer (100k transitions)
    - Soft target updates (tau=0.005)
    """

    def __init__(
        self,
        state_dim=14,
        action_dim=3,
        hidden_size=256,
        lr=3e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=50000,
        batch_size=64,
        buffer_size=100000,
        target_update_tau=0.005,
        device="auto",
    ):
        # Auto-detect GPU (gracefully fall back to CPU)
        if device in ("auto", "cuda"):
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"  DQN Agent using: CUDA ({torch.cuda.get_device_name(0)})")
            else:
                self.device = torch.device("cpu")
                print(f"  DQN Agent using: CPU (CUDA not available — still fast for this model)")
        else:
            self.device = torch.device(device)
            print(f"  DQN Agent using device: {self.device}")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = target_update_tau

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Tracking
        self.training_losses = []

    @property
    def epsilon(self):
        """Current exploration rate (exponential decay)."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-self.steps_done / self.epsilon_decay)

    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: numpy array of shape (state_dim,)
            training: if False, use greedy policy (no exploration)

        Returns:
            action: integer action index
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """
        Perform one gradient step on a batch from the replay buffer.

        Returns:
            loss value (float) or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        self.steps_done += 1

        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values: Q(s, a)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft update target network
        self._soft_update()

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        return loss_val

    def _soft_update(self):
        """Soft update target network: θ_target = τ*θ + (1-τ)*θ_target"""
        for target_param, param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "epsilon": self.epsilon,
        }, path)
        print(f"  Model saved to {path}")

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
        print(f"  Model loaded from {path} (step {self.steps_done})")

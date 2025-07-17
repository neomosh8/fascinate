# rl_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Tuple, List
import pickle
import os


class DQN(nn.Module):
    """Deep Q-Network for strategy selection"""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.network(x)


class EngagementRL:
    """RL Agent for optimizing communication strategies"""

    def __init__(self, strategy_count: int, state_size: int = 10):
        self.strategy_count = strategy_count
        self.state_size = state_size

        # Hyperparameters
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.8  # discount factor

        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_size, strategy_count).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Experience replay
        self.memory = deque(maxlen=2000)
        self.batch_size = 32

        # State tracking
        self.current_state = np.zeros(state_size)
        self.last_action = None
        self.engagement_history = deque(maxlen=20)

        # Performance tracking
        self.episode_rewards = []
        self.strategy_performance = {}

    def get_state(self, current_engagement: float, engagement_change: float,
                  conversation_length: int) -> np.ndarray:
        """Create state vector for RL agent"""
        state = np.zeros(self.state_size)

        # Current metrics
        state[0] = current_engagement
        state[1] = engagement_change
        state[2] = conversation_length / 100.0  # normalize

        # Engagement history features
        if len(self.engagement_history) > 0:
            state[3] = np.mean(self.engagement_history)  # avg engagement
            state[4] = np.std(self.engagement_history)  # engagement variance
            state[5] = max(self.engagement_history)  # peak engagement
            state[6] = min(self.engagement_history)  # lowest engagement

        # Recent trend
        if len(self.engagement_history) >= 5:
            recent = list(self.engagement_history)[-5:]
            state[7] = (recent[-1] - recent[0]) / 5.0  # trend slope

        # Strategy performance context
        if self.last_action is not None:
            state[8] = self.strategy_performance.get(self.last_action, 0.0)

        # Time of day effect (simple)
        import time
        state[9] = (time.time() % 86400) / 86400.0  # normalized time of day

        return state

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.strategy_count - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def calculate_reward(self, engagement_change: float, current_engagement: float) -> float:
        """Calculate reward based on engagement metrics"""
        # Primary reward: engagement improvement
        reward = engagement_change * 10.0

        # Bonus for maintaining high engagement
        if current_engagement > 0.7:
            reward += 2.0

        # Penalty for very low engagement
        if current_engagement < 0.3:
            reward -= 1.0

        return reward

    def update_performance(self, action: int, reward: float):
        """Update strategy performance tracking"""
        if action not in self.strategy_performance:
            self.strategy_performance[action] = 0.0

        # Exponential moving average
        alpha = 0.1
        self.strategy_performance[action] = (
                alpha * reward + (1 - alpha) * self.strategy_performance[action]
        )

    def train_step(self):
        """Train the neural network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'strategy_performance': self.strategy_performance,
            'episode_rewards': self.episode_rewards
        }, filepath)

    def load_model(self, filepath: str):
        """Load a trained model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.strategy_performance = checkpoint.get('strategy_performance', {})
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            print(f"Model loaded from {filepath}")
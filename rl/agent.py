"""Tabular Q-learning agent for communication strategy selection."""

import numpy as np
from typing import Dict, Tuple, Optional
import pickle
from pathlib import Path

from config import RLConfig
from .strategy import StrategySpace, Strategy


class QLearningAgent:
    """Tabular Q-learning agent."""

    def __init__(self, strategy_space: StrategySpace, config: RLConfig = RLConfig()):
        self.strategy_space = strategy_space
        self.config = config

        # Initialize Q-table
        # State: (last_strategy_index, last_attention_bucket, user_spoke)
        # - last_strategy_index: 0 to num_strategies-1
        # - last_attention_bucket: 0-9 (discretized 0.0-1.0)
        # - user_spoke: 0 or 1
        self.num_states = strategy_space.num_strategies * 10 * 2
        self.num_actions = strategy_space.num_strategies

        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.epsilon = config.epsilon_initial
        self.step_count = 0

    def state_to_index(self, last_strategy_idx: int, attention: float, user_spoke: bool) -> int:
        """Convert state tuple to Q-table index."""
        attention_bucket = min(int(attention * 10), 9)
        user_spoke_int = 1 if user_spoke else 0

        return (last_strategy_idx * 10 * 2 +
                attention_bucket * 2 +
                user_spoke_int)

    def choose_action(self, state_idx: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            return np.argmax(self.q_table[state_idx])

    def update(self, state_idx: int, action: int, reward: float, next_state_idx: int):
        """Update Q-value using Q-learning formula."""
        old_q = self.q_table[state_idx, action]
        max_next_q = np.max(self.q_table[next_state_idx])

        new_q = old_q + self.config.learning_rate * (
                reward + self.config.discount_factor * max_next_q - old_q
        )

        self.q_table[state_idx, action] = new_q

        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )
        self.step_count += 1

    def save(self, filepath: Path):
        """Save Q-table and agent state."""
        state = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath: Path):
        """Load Q-table and agent state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.q_table = state['q_table']
        self.epsilon = state['epsilon']
        self.step_count = state['step_count']
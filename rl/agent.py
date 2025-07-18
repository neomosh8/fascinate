"""Tabular Q-learning agent for communication strategy selection."""

import numpy as np
from typing import Dict, Tuple, Optional, List
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
        self.num_states = strategy_space.num_strategies * 10 * 2
        self.num_actions = strategy_space.num_strategies

        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.epsilon = config.epsilon_initial
        self.step_count = 0

        # Performance tracking
        self.strategy_usage_count = np.zeros(self.num_actions)
        self.strategy_rewards = np.zeros(self.num_actions)
        self.strategy_total_rewards = np.zeros(self.num_actions)
        self.total_reward = 0.0
        self.reward_history = []

    def state_to_index(self, last_strategy_idx: int, attention: float, user_spoke: bool) -> int:
        """Convert state tuple to Q-table index."""
        attention_bucket = min(int(attention * 10), 9)
        user_spoke_int = 1 if user_spoke else 0

        return (last_strategy_idx * 10 * 2 +
                attention_bucket * 2 +
                user_spoke_int)

    def choose_action(self, state_idx: int) -> int:
        """Epsilon-greedy action selection with random tie-breaking."""
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        else:
            max_q = np.max(self.q_table[state_idx])
            max_actions = np.where(self.q_table[state_idx] == max_q)[0]
            action = np.random.choice(max_actions)

        # Track usage
        self.strategy_usage_count[action] += 1
        return action

    def update(self, state_idx: int, action: int, reward: float, next_state_idx: int):
        """Update Q-value using Q-learning formula."""
        old_q = self.q_table[state_idx, action]
        max_next_q = np.max(self.q_table[next_state_idx])

        new_q = old_q + self.config.learning_rate * (
                reward + self.config.discount_factor * max_next_q - old_q
        )

        self.q_table[state_idx, action] = new_q

        # Track performance
        self.strategy_total_rewards[action] += reward
        self.total_reward += reward
        self.reward_history.append(reward)

        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay
        )
        self.step_count += 1

    def get_best_strategy(self) -> Tuple[int, float]:
        """Get the overall best performing strategy."""
        # Calculate average reward per strategy
        avg_rewards = np.zeros(self.num_actions)
        for i in range(self.num_actions):
            if self.strategy_usage_count[i] > 0:
                avg_rewards[i] = self.strategy_total_rewards[i] / self.strategy_usage_count[i]

        best_strategy_idx = np.argmax(avg_rewards)
        best_performance = avg_rewards[best_strategy_idx]

        return best_strategy_idx, best_performance

    def get_top_strategies(self, top_k: int = 5) -> List[Tuple[int, float, int]]:
        """Get top k strategies by average reward."""
        strategy_stats = []

        for i in range(self.num_actions):
            if self.strategy_usage_count[i] > 0:
                avg_reward = self.strategy_total_rewards[i] / self.strategy_usage_count[i]
                usage_count = int(self.strategy_usage_count[i])
                strategy_stats.append((i, avg_reward, usage_count))

        # Sort by average reward (descending)
        strategy_stats.sort(key=lambda x: x[1], reverse=True)

        return strategy_stats[:top_k]

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        if self.step_count == 0:
            return {"error": "No data available"}

        best_strategy_idx, best_performance = self.get_best_strategy()
        top_strategies = self.get_top_strategies()

        # Calculate learning curve metrics
        recent_rewards = self.reward_history[-50:] if len(self.reward_history) >= 50 else self.reward_history
        recent_avg = np.mean(recent_rewards) if recent_rewards else 0

        early_rewards = self.reward_history[:50] if len(self.reward_history) >= 50 else self.reward_history[:len(self.reward_history)//2]
        early_avg = np.mean(early_rewards) if early_rewards else 0

        return {
            "total_turns": self.step_count,
            "total_reward": self.total_reward,
            "average_reward": self.total_reward / self.step_count,
            "best_strategy": {
                "index": best_strategy_idx,
                "strategy": self.strategy_space.get_strategy(best_strategy_idx),
                "average_reward": best_performance,
                "usage_count": int(self.strategy_usage_count[best_strategy_idx])
            },
            "top_strategies": [
                {
                    "index": idx,
                    "strategy": self.strategy_space.get_strategy(idx),
                    "average_reward": avg_reward,
                    "usage_count": usage_count
                }
                for idx, avg_reward, usage_count in top_strategies
            ],
            "learning_progress": {
                "early_average_reward": early_avg,
                "recent_average_reward": recent_avg,
                "improvement": recent_avg - early_avg
            },
            "exploration_stats": {
                "final_epsilon": self.epsilon,
                "strategies_tried": int(np.sum(self.strategy_usage_count > 0)),
                "total_strategies": self.num_actions
            }
        }

    def save(self, filepath: Path):
        """Save Q-table and agent state."""
        state = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'strategy_usage_count': self.strategy_usage_count,
            'strategy_total_rewards': self.strategy_total_rewards,
            'total_reward': self.total_reward,
            'reward_history': self.reward_history
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
        self.strategy_usage_count = state.get('strategy_usage_count', np.zeros(self.num_actions))
        self.strategy_total_rewards = state.get('strategy_total_rewards', np.zeros(self.num_actions))
        self.total_reward = state.get('total_reward', 0.0)
        self.reward_history = state.get('reward_history', [])
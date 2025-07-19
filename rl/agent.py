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

    def choose_action_ucb(self, state_idx: int, c: float = 2.0) -> int:
        """UCB action selection - better for exploration-exploitation balance."""
        if self.step_count == 0:
            return np.random.randint(0, self.num_actions)

        ucb_values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            usage_count = max(self.strategy_usage_count[action], 1)
            avg_reward = self.strategy_total_rewards[action] / usage_count
            confidence_interval = c * np.sqrt(np.log(self.step_count) / usage_count)
            ucb_values[action] = avg_reward + confidence_interval

        return int(np.argmax(ucb_values))

    def choose_action(self, state_idx: int, use_ucb: bool = True) -> int:
        """Choose action with either UCB or epsilon-greedy."""
        if use_ucb and self.step_count > 0:
            action = self.choose_action_ucb(state_idx, self.config.ucb_confidence)
        else:
            if np.random.random() < self.epsilon:
                action = np.random.randint(0, self.num_actions)
            else:
                max_q = np.max(self.q_table[state_idx])
                max_actions = np.where(self.q_table[state_idx] == max_q)[0]
                action = int(np.random.choice(max_actions))

        # Track usage
        self.strategy_usage_count[action] += 1
        return int(action)

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

    def get_best_strategy(self, min_usage: int = 3) -> Tuple[int, float]:
        """Get the overall best performing strategy with minimum usage requirement."""
        valid_strategies = []
        for i in range(self.num_actions):
            usage_count = int(self.strategy_usage_count[i])
            if usage_count >= min_usage:
                avg_reward = self.strategy_total_rewards[i] / usage_count
                valid_strategies.append((i, avg_reward))

        if not valid_strategies:
            best_idx = int(np.argmax(self.strategy_usage_count))
            avg_reward = (
                self.strategy_total_rewards[best_idx] /
                max(self.strategy_usage_count[best_idx], 1)
            )
            return best_idx, avg_reward

        best_strategy = max(valid_strategies, key=lambda x: x[1])
        return best_strategy[0], best_strategy[1]

    def get_top_strategies(self, top_k: int = 5, min_usage: int = 3) -> List[Tuple[int, float, int]]:
        """Get top k strategies by average reward with minimum usage requirement."""
        strategy_stats = []

        for i in range(self.num_actions):
            usage_count = int(self.strategy_usage_count[i])
            if usage_count >= min_usage:
                avg_reward = self.strategy_total_rewards[i] / usage_count
                strategy_stats.append((i, avg_reward, usage_count))

        # Sort by average reward (descending)
        strategy_stats.sort(key=lambda x: x[1], reverse=True)

        return strategy_stats[:top_k]

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary with statistical confidence."""
        if self.step_count == 0:
            return {"error": "No data available"}

        min_usage = max(3, self.step_count // 20)

        try:
            best_strategy_idx, best_performance = self.get_best_strategy(min_usage)
            top_strategies = self.get_top_strategies(min_usage=min_usage)

            confident_strategies = []
            for idx, avg_reward, usage_count in top_strategies:
                rewards = [self.strategy_total_rewards[idx] / usage_count] * usage_count
                std_error = np.sqrt(np.var(rewards) / usage_count) if usage_count > 1 else 0
                confidence = 1.96 * std_error
                confident_strategies.append({
                    "index": idx,
                    "strategy": self.strategy_space.get_strategy(idx),
                    "average_reward": avg_reward,
                    "usage_count": usage_count,
                    "confidence_interval": confidence,
                    "statistical_significance": "high" if usage_count >= min_usage else "low",
                })
        except Exception:
            best_strategy_idx = int(np.argmax(self.strategy_usage_count))
            best_performance = (
                self.strategy_total_rewards[best_strategy_idx] /
                max(self.strategy_usage_count[best_strategy_idx], 1)
            )
            confident_strategies = []
            top_strategies = []

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
            },
            "statistically_significant_strategies": confident_strategies,
            "minimum_usage_threshold": min_usage,
            "exploration_note": f"Strategies with <{min_usage} uses may not be reliable"
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

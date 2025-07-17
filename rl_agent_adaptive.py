# rl_agent_adaptive.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Tuple, List
import pickle
import os


class AdaptiveEngagementRL:
    """RL Agent that adapts exploration/exploitation based on recent performance"""

    def __init__(self, strategy_count: int, state_size: int = 10):
        self.strategy_count = strategy_count
        self.state_size = state_size

        print(f"🤖 Adaptive RL Agent initialized with {strategy_count} strategies")

        # Dynamic exploration parameters
        self.base_epsilon = 0.3  # Baseline exploration
        self.current_epsilon = 0.7  # Start higher
        self.epsilon_min = 0.05
        self.epsilon_max = 0.8

        # Performance tracking for adaptation
        self.recent_rewards = deque(maxlen=5)  # Last 5 rewards
        self.recent_engagements = deque(maxlen=5)  # Last 5 engagement levels
        self.success_threshold = 0.7  # What we consider "high engagement"
        self.poor_threshold = 0.4  # What we consider "poor engagement"

        # Strategy success tracking
        self.strategy_performance = {}  # Average reward per strategy
        self.strategy_success_rate = {}  # Success rate (% above threshold)
        self.strategy_usage_count = {}
        self.strategy_recent_performance = {}  # Recent performance per strategy

        # Hot strategies (currently working well)
        self.hot_strategies = set()  # Strategies that are working well
        self.cold_strategies = set()  # Strategies that aren't working

        # Neural network setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_size, strategy_count).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.002)  # Faster learning

        # Experience replay
        self.memory = deque(maxlen=1000)  # Smaller buffer for faster adaptation
        self.batch_size = 16  # Smaller batches for faster learning
        self.gamma = 0.9  # Higher discount for immediate rewards

        # State tracking
        self.current_state = np.zeros(state_size)
        self.last_action = None
        self.last_reward = None
        self.engagement_history = deque(maxlen=20)

    def select_action(self, state: np.ndarray) -> int:
        """Adaptive action selection - exploits good strategies immediately"""

        # Adapt epsilon based on recent performance
        self._adapt_epsilon()

        # If we have hot strategies and we're in exploitation mode, favor them
        if self.hot_strategies and random.random() > self.current_epsilon:
            # Exploitation: prefer hot strategies
            if random.random() < 0.7:  # 70% chance to pick from hot strategies
                action = self._select_from_hot_strategies(state)
            else:
                action = self._select_with_q_network(state)
        else:
            # Exploration: try new strategies or Q-network
            if random.random() < 0.5:
                action = self._explore_new_strategy()
            else:
                action = self._select_with_q_network(state)

        # Update usage count
        self.strategy_usage_count[action] = self.strategy_usage_count.get(action, 0) + 1

        # Ensure valid range
        action = action % self.strategy_count

        # Show selection reasoning
        hot_count = len(self.hot_strategies)
        cold_count = len(self.cold_strategies)
        selection_type = "🔥HOT" if action in self.hot_strategies else "🆕NEW" if action not in self.strategy_usage_count or \
                                                                                self.strategy_usage_count[
                                                                                    action] < 3 else "🎯NET"

        print(
            f"🎯 {selection_type} strategy {action} (ε={self.current_epsilon:.3f}, hot={hot_count}, cold={cold_count})")

        return action

    def _adapt_epsilon(self):
        """Dynamically adapt epsilon based on recent performance"""
        if len(self.recent_engagements) < 3:
            return

        # Calculate recent performance metrics
        recent_avg = np.mean(list(self.recent_engagements))
        recent_trend = (list(self.recent_engagements)[-1] - list(self.recent_engagements)[0]) if len(
            self.recent_engagements) > 1 else 0
        recent_reward_avg = np.mean(list(self.recent_rewards)) if self.recent_rewards else 0

        # Adaptation logic
        if recent_avg > self.success_threshold and recent_trend >= 0:
            # Things are going well - exploit more
            self.current_epsilon = max(self.epsilon_min, self.current_epsilon * 0.8)
            mode = "EXPLOIT"
        elif recent_avg < self.poor_threshold or recent_trend < -0.1:
            # Things are going poorly - explore more
            self.current_epsilon = min(self.epsilon_max, self.current_epsilon * 1.3)
            mode = "EXPLORE"
        else:
            # Moderate performance - slight trend toward baseline
            target = self.base_epsilon
            self.current_epsilon = 0.9 * self.current_epsilon + 0.1 * target
            mode = "BALANCE"

        print(f"📈 Adaptation: {mode} (avg={recent_avg:.2f}, trend={recent_trend:+.2f}) → ε={self.current_epsilon:.3f}")

    def _select_from_hot_strategies(self, state: np.ndarray) -> int:
        """Select from strategies that are currently working well"""
        if not self.hot_strategies:
            return self._select_with_q_network(state)

        # Weight hot strategies by their recent performance
        hot_list = list(self.hot_strategies)
        weights = []

        for action in hot_list:
            performance = self.strategy_recent_performance.get(action, 0.5)
            weights.append(max(0.1, performance))  # Ensure positive weight

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Select based on performance weights
        selected_idx = np.random.choice(len(hot_list), p=weights)
        return hot_list[selected_idx]

    def _select_with_q_network(self, state: np.ndarray) -> int:
        """Select using Q-network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def _explore_new_strategy(self) -> int:
        """Explore strategies that haven't been tried much"""
        # Prefer strategies with low usage count
        usage_counts = [self.strategy_usage_count.get(i, 0) for i in range(self.strategy_count)]
        min_usage = min(usage_counts)

        # Find strategies with minimal usage
        candidates = [i for i in range(self.strategy_count) if usage_counts[i] <= min_usage + 2]

        # Remove cold strategies from candidates
        candidates = [i for i in candidates if i not in self.cold_strategies]

        if candidates:
            return random.choice(candidates)
        else:
            return random.randint(0, self.strategy_count - 1)

    def update_performance(self, action: int, reward: float, engagement: float):
        """Update performance and classify strategies as hot/cold"""
        action = action % self.strategy_count

        # Update recent tracking
        self.recent_rewards.append(reward)
        self.recent_engagements.append(engagement)

        # Update strategy performance
        if action not in self.strategy_performance:
            self.strategy_performance[action] = reward
            self.strategy_success_rate[action] = 1.0 if engagement > self.success_threshold else 0.0
            self.strategy_recent_performance[action] = engagement
        else:
            # Exponential moving average with higher weight on recent performance
            alpha = 0.4  # Give more weight to recent performance
            self.strategy_performance[action] = (
                    alpha * reward + (1 - alpha) * self.strategy_performance[action]
            )

            # Update success rate
            old_success = self.strategy_success_rate[action]
            is_success = 1.0 if engagement > self.success_threshold else 0.0
            self.strategy_success_rate[action] = (
                    alpha * is_success + (1 - alpha) * old_success
            )

            # Recent performance (faster adaptation)
            self.strategy_recent_performance[action] = (
                    0.6 * engagement + 0.4 * self.strategy_recent_performance[action]
            )

        # Classify as hot or cold
        self._classify_strategy(action)

        # Show immediate feedback
        status = "🔥HOT" if action in self.hot_strategies else "🥶COLD" if action in self.cold_strategies else "🔄EVAL"
        print(
            f"🎯 Strategy {action} → {status} (perf={self.strategy_recent_performance[action]:.2f}, success={self.strategy_success_rate[action]:.2f})")

    def _classify_strategy(self, action: int):
        """Classify strategy as hot, cold, or neutral"""
        usage_count = self.strategy_usage_count.get(action, 0)

        # Need at least 2 uses to classify
        if usage_count < 2:
            return

        recent_perf = self.strategy_recent_performance.get(action, 0.5)
        success_rate = self.strategy_success_rate.get(action, 0.5)

        # Hot strategy criteria: high recent performance AND good success rate
        if recent_perf > self.success_threshold and success_rate > 0.6:
            self.hot_strategies.add(action)
            self.cold_strategies.discard(action)

        # Cold strategy criteria: poor recent performance OR low success rate
        elif recent_perf < self.poor_threshold or success_rate < 0.3:
            self.cold_strategies.add(action)
            self.hot_strategies.discard(action)

        # Neutral: remove from both if neither hot nor cold
        elif self.poor_threshold <= recent_perf <= self.success_threshold:
            self.hot_strategies.discard(action)
            self.cold_strategies.discard(action)

    def train_step(self):
        """Fast training for immediate adaptation"""
        if len(self.memory) < self.batch_size:
            return

        # Train more frequently with smaller batches
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
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

    def get_strategy_stats(self):
        """Get comprehensive strategy statistics"""
        total_usage = sum(self.strategy_usage_count.values())

        return {
            'total_strategies': self.strategy_count,
            'strategies_tried': len(self.strategy_usage_count),
            'hot_strategies': len(self.hot_strategies),
            'cold_strategies': len(self.cold_strategies),
            'current_epsilon': self.current_epsilon,
            'recent_avg_engagement': np.mean(list(self.recent_engagements)) if self.recent_engagements else 0,
            'hot_list': list(self.hot_strategies)[:5],  # Show top 5
            'total_usage': total_usage
        }

    # Keep the rest of the methods (save_model, load_model, etc.) the same...
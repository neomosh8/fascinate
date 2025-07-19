"""Hierarchical Multi-Armed Bandit with Faster Convergence."""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass
import time

from config import TONES, TOPICS, EMOTIONS, HOOKS
from .strategy import Strategy


@dataclass
class ComponentPerformance:
    """Track performance statistics for a component."""
    successes: int = 0
    failures: int = 0
    total_reward: float = 0.0
    usage_count: int = 0
    recent_rewards: deque = None

    def __post_init__(self):
        if self.recent_rewards is None:
            self.recent_rewards = deque(maxlen=10)  # Track recent performance

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.5
        return self.successes / self.usage_count

    @property
    def average_reward(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.total_reward / self.usage_count

    @property
    def recent_average(self) -> float:
        """Recent performance average for faster adaptation."""
        if len(self.recent_rewards) == 0:
            return 0.0
        return np.mean(self.recent_rewards)


class FastThompsonSamplingBandit:
    """Thompson Sampling bandit optimized for faster convergence."""

    def __init__(self, arms: List[str],
                 alpha_prior: float = 2.0,  # More optimistic prior
                 beta_prior: float = 1.0,   # Asymmetric prior favoring success
                 learning_rate: float = 0.3):  # Faster adaptation
        self.arms = arms
        self.num_arms = len(arms)
        self.learning_rate = learning_rate

        # More optimistic Beta priors for faster convergence
        self.alpha = np.full(self.num_arms, alpha_prior)
        self.beta = np.full(self.num_arms, beta_prior)

        # Performance tracking with recent bias
        self.performance = {arm: ComponentPerformance() for arm in arms}
        self.arm_to_index = {arm: i for i, arm in enumerate(arms)}

        # Track exploration vs exploitation
        self.total_selections = 0
        self.exploration_bonus = 0.1  # Bonus for less-tried arms

    def select_arm(self) -> str:
        """Select arm using enhanced Thompson Sampling with exploration bonus."""
        self.total_selections += 1

        # Sample from Beta distribution for each arm
        samples = np.random.beta(self.alpha, self.beta)

        # Add exploration bonus for under-explored arms
        if self.total_selections < 20:  # Early exploration phase
            for i, arm in enumerate(self.arms):
                usage_count = self.performance[arm].usage_count
                if usage_count < 3:  # Boost under-explored arms
                    samples[i] += self.exploration_bonus * (3 - usage_count)

        selected_index = np.argmax(samples)
        return self.arms[selected_index]

    def update(self, arm: str, reward: float):
        """Update bandit with continuous reward values (not just binary)."""
        idx = self.arm_to_index[arm]
        perf = self.performance[arm]

        # Update performance tracking
        perf.usage_count += 1
        perf.total_reward += reward
        perf.recent_rewards.append(reward)

        # Continuous reward update (better than binary)
        # Map reward [-2, 2] to success probability
        normalized_reward = np.clip((reward + 2) / 4, 0, 1)  # Map to [0,1]

        # Use weighted update for faster adaptation
        effective_reward = normalized_reward
        if len(perf.recent_rewards) >= 3:
            # Boost if recent trend is positive
            recent_trend = np.mean(list(perf.recent_rewards)[-3:]) - np.mean(list(perf.recent_rewards)[:-3]) if len(perf.recent_rewards) > 3 else 0
            if recent_trend > 0:
                effective_reward = min(1.0, effective_reward + 0.1)

        # Faster Beta updates
        update_strength = self.learning_rate * 5  # Stronger updates

        if effective_reward > 0.5:
            perf.successes += 1
            self.alpha[idx] += effective_reward * update_strength
        else:
            perf.failures += 1
            self.beta[idx] += (1 - effective_reward) * update_strength

        # Prevent parameters from getting too extreme
        self.alpha[idx] = np.clip(self.alpha[idx], 0.5, 20)
        self.beta[idx] = np.clip(self.beta[idx], 0.5, 20)

    def get_best_arm(self) -> Tuple[str, float]:
        """Get the arm with highest expected reward."""
        best_arm = max(self.arms, key=lambda arm: self.performance[arm].recent_average
                      if len(self.performance[arm].recent_rewards) >= 3
                      else self.performance[arm].average_reward)
        return best_arm, self.performance[best_arm].average_reward

    def get_confidence_intervals(self) -> Dict[str, Tuple[float, float]]:
        """Get 95% confidence intervals for each arm."""
        intervals = {}
        for i, arm in enumerate(self.arms):
            # 95% confidence interval for Beta distribution
            samples = np.random.beta(self.alpha[i], self.beta[i], 1000)
            intervals[arm] = (np.percentile(samples, 2.5), np.percentile(samples, 97.5))
        return intervals


class FasterRestartManager:
    """More conservative restart strategy that doesn't disrupt learning."""

    def __init__(self,
                 window_size: int = 15,  # Smaller window for faster detection
                 stagnation_threshold: float = 0.02,  # Lower threshold
                 min_performance_threshold: float = 0.2,  # Lower threshold
                 restart_cooldown: int = 100):  # Longer cooldown to avoid disruption

        self.window_size = window_size
        self.stagnation_threshold = stagnation_threshold
        self.min_performance_threshold = min_performance_threshold
        self.restart_cooldown = restart_cooldown

        # Performance tracking
        self.reward_history = deque(maxlen=window_size * 3)
        self.last_restart_step = 0
        self.step_count = 0

        # More conservative restart strategies
        self.restart_strategies = ['gentle_reset', 'exploration_boost']
        self.restart_performance = defaultdict(list)

    def should_restart(self, recent_rewards: List[float]) -> bool:
        """More conservative restart criteria."""
        self.step_count += 1
        self.reward_history.extend(recent_rewards)

        # Don't restart too frequently or too early
        if (self.step_count - self.last_restart_step < self.restart_cooldown or
            self.step_count < 30):  # Wait longer before first restart
            return False

        if len(self.reward_history) < self.window_size:
            return False

        # Only restart if performance is genuinely poor
        recent_window = list(self.reward_history)[-self.window_size:]
        recent_avg = np.mean(recent_window)

        # More conservative restart condition
        severe_underperformance = recent_avg < self.min_performance_threshold

        return severe_underperformance

    def execute_restart(self, bandits: Dict[str, FastThompsonSamplingBandit]) -> str:
        """Execute gentle restart strategy."""
        self.last_restart_step = self.step_count

        # Choose gentler restart strategy
        strategy = 'gentle_reset'  # Always use gentle approach

        if strategy == 'gentle_reset':
            self._gentle_reset(bandits)
        else:  # exploration_boost
            self._boost_exploration(bandits)

        return strategy

    def _gentle_reset(self, bandits: Dict[str, FastThompsonSamplingBandit]):
        """Gently reset only the worst performing arms."""
        for bandit in bandits.values():
            # Only reset arms with very poor performance
            for arm in bandit.arms:
                perf = bandit.performance[arm]
                if perf.usage_count >= 5 and perf.average_reward < 0:
                    # Gentle reset - don't fully reset, just reduce confidence
                    idx = bandit.arm_to_index[arm]
                    bandit.alpha[idx] = max(bandit.alpha[idx] * 0.7, 2.0)
                    bandit.beta[idx] = max(bandit.beta[idx] * 0.7, 1.0)

    def _boost_exploration(self, bandits: Dict[str, FastThompsonSamplingBandit]):
        """Boost exploration without destroying learned preferences."""
        for bandit in bandits.values():
            # Slightly reduce confidence to encourage more exploration
            bandit.alpha *= 0.9
            bandit.beta *= 0.9
            bandit.alpha = np.maximum(bandit.alpha, 1.5)
            bandit.beta = np.maximum(bandit.beta, 0.8)


class HierarchicalBanditAgent:
    """Faster-converging hierarchical bandit agent."""

    def __init__(self):
        # Component-level bandits with faster convergence
        self.tone_bandit = FastThompsonSamplingBandit(TONES)
        self.topic_bandit = FastThompsonSamplingBandit(TOPICS)
        self.emotion_bandit = FastThompsonSamplingBandit(EMOTIONS)
        self.hook_bandit = FastThompsonSamplingBandit(HOOKS)

        self.bandits = {
            'tone': self.tone_bandit,
            'topic': self.topic_bandit,
            'emotion': self.emotion_bandit,
            'hook': self.hook_bandit
        }

        # More conservative restart manager
        self.restart_manager = FasterRestartManager()

        # Strategy tracking
        self.strategy_history = []
        self.recent_rewards = deque(maxlen=15)  # Smaller window
        self.step_count = 0

        # Early learning boost
        self.early_learning_phase = True
        self.early_learning_cutoff = 25  # First 25 turns get boosted learning

    def select_strategy(self) -> Strategy:
        """Select strategy with early learning boost."""
        # Select components
        tone = self.tone_bandit.select_arm()
        topic = self.topic_bandit.select_arm()
        emotion = self.emotion_bandit.select_arm()
        hook = self.hook_bandit.select_arm()

        strategy = Strategy(
            tone=tone,
            topic=topic,
            emotion=emotion,
            hook=hook,
            index=self.step_count
        )

        self.strategy_history.append(strategy)
        self.step_count += 1

        # Check if still in early learning phase
        if self.step_count >= self.early_learning_cutoff:
            self.early_learning_phase = False

        return strategy

    def update(self, strategy: Strategy, reward: float):
        """Update bandits with enhanced learning during early phase."""
        # Normalize reward to reasonable range
        normalized_reward = np.clip(reward, -2.0, 2.0)

        # Early learning boost - amplify rewards to accelerate learning
        if self.early_learning_phase:
            # Amplify positive rewards more than negative ones
            if normalized_reward > 0:
                boosted_reward = normalized_reward * 1.5
            else:
                boosted_reward = normalized_reward * 1.2
            normalized_reward = np.clip(boosted_reward, -2.0, 2.0)

        # Update component bandits
        self.tone_bandit.update(strategy.tone, normalized_reward)
        self.topic_bandit.update(strategy.topic, normalized_reward)
        self.emotion_bandit.update(strategy.emotion, normalized_reward)
        self.hook_bandit.update(strategy.hook, normalized_reward)

        # Track recent performance
        self.recent_rewards.append(normalized_reward)

        # Less aggressive restart checking
        if (len(self.recent_rewards) >= 10 and
            self.step_count > 30 and  # Wait longer before allowing restarts
            self.restart_manager.should_restart([normalized_reward])):

            restart_strategy = self.restart_manager.execute_restart(self.bandits)
            print(f"🔄 Conservative restart executed: {restart_strategy} (step {self.step_count})")

            # Track restart performance more conservatively
            if len(self.recent_rewards) >= 8:
                window_before = list(self.recent_rewards)[-8:-4]
                if window_before and np.mean(window_before) < 0.1:  # Only if truly poor
                    self.restart_manager.restart_performance[restart_strategy].extend(window_before)

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        summary = {
            'step_count': self.step_count,
            'early_learning_phase': self.early_learning_phase,
            'recent_performance': list(self.recent_rewards)[-10:],
            'average_recent_reward': np.mean(list(self.recent_rewards)[-10:]) if self.recent_rewards else 0,
            'components': {}
        }

        # Component performance with faster metrics
        for component_name, bandit in self.bandits.items():
            best_arm, best_score = bandit.get_best_arm()
            confidence_intervals = bandit.get_confidence_intervals()

            summary['components'][component_name] = {
                'best_choice': best_arm,
                'best_score': best_score,
                'confidence_intervals': confidence_intervals,
                'usage_stats': {
                    arm: {
                        'usage_count': bandit.performance[arm].usage_count,
                        'average_reward': bandit.performance[arm].average_reward,
                        'recent_average': bandit.performance[arm].recent_average,
                        'success_rate': bandit.performance[arm].success_rate
                    }
                    for arm in bandit.arms
                }
            }

        # Restart statistics
        summary['restart_stats'] = {
            'total_restarts': sum(len(perfs) for perfs in self.restart_manager.restart_performance.values()),
            'last_restart_step': self.restart_manager.last_restart_step,
            'restart_strategy_performance': dict(self.restart_manager.restart_performance)
        }

        return summary

    def save(self, filepath: Path):
        """Save bandit state."""
        state = {
            'step_count': self.step_count,
            'early_learning_phase': self.early_learning_phase,
            'bandits': {},
            'restart_manager': {
                'reward_history': list(self.restart_manager.reward_history),
                'last_restart_step': self.restart_manager.last_restart_step,
                'step_count': self.restart_manager.step_count,
                'restart_performance': dict(self.restart_manager.restart_performance)
            }
        }

        # Save bandit states
        for name, bandit in self.bandits.items():
            state['bandits'][name] = {
                'alpha': bandit.alpha.tolist(),
                'beta': bandit.beta.tolist(),
                'total_selections': bandit.total_selections,
                'performance': {}
            }

            # Save performance data
            for arm, perf in bandit.performance.items():
                state['bandits'][name]['performance'][arm] = {
                    'successes': perf.successes,
                    'failures': perf.failures,
                    'total_reward': perf.total_reward,
                    'usage_count': perf.usage_count,
                    'recent_rewards': list(perf.recent_rewards)
                }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load(self, filepath: Path):
        """Load bandit state."""
        if not filepath.exists():
            return

        with open(filepath, 'r') as f:
            state = json.load(f)

        self.step_count = state.get('step_count', 0)
        self.early_learning_phase = state.get('early_learning_phase', True)

        # Load bandit states
        for name, bandit_state in state.get('bandits', {}).items():
            if name in self.bandits:
                bandit = self.bandits[name]
                bandit.alpha = np.array(bandit_state['alpha'])
                bandit.beta = np.array(bandit_state['beta'])
                bandit.total_selections = bandit_state.get('total_selections', 0)

                for arm, perf_data in bandit_state.get('performance', {}).items():
                    if arm in bandit.performance:
                        perf = bandit.performance[arm]
                        perf.successes = perf_data['successes']
                        perf.failures = perf_data['failures']
                        perf.total_reward = perf_data['total_reward']
                        perf.usage_count = perf_data['usage_count']
                        perf.recent_rewards = deque(perf_data.get('recent_rewards', []), maxlen=10)

        # Load restart manager state
        restart_state = state.get('restart_manager', {})
        self.restart_manager.reward_history = deque(
            restart_state.get('reward_history', []),
            maxlen=self.restart_manager.window_size * 3
        )
        self.restart_manager.last_restart_step = restart_state.get('last_restart_step', 0)
        self.restart_manager.step_count = restart_state.get('step_count', 0)
        self.restart_manager.restart_performance = defaultdict(
            list, restart_state.get('restart_performance', {})
        )
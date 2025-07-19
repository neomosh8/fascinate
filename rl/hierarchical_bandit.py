"""Hierarchical Multi-Armed Bandit for Strategy Optimization."""

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
    
    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.5  # Prior belief
        return self.successes / self.usage_count
    
    @property
    def average_reward(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.total_reward / self.usage_count


class ThompsonSamplingBandit:
    """Thompson Sampling bandit for individual components."""
    
    def __init__(self, arms: List[str], alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.arms = arms
        self.num_arms = len(arms)
        
        # Beta distribution parameters for Thompson Sampling
        self.alpha = np.full(self.num_arms, alpha_prior)
        self.beta = np.full(self.num_arms, beta_prior)
        
        # Performance tracking
        self.performance = {arm: ComponentPerformance() for arm in arms}
        self.arm_to_index = {arm: i for i, arm in enumerate(arms)}
        
    def select_arm(self) -> str:
        """Select arm using Thompson Sampling."""
        # Sample from Beta distribution for each arm
        samples = np.random.beta(self.alpha, self.beta)
        selected_index = np.argmax(samples)
        return self.arms[selected_index]
    
    def update(self, arm: str, reward: float):
        """Update bandit with reward (assumes reward in [0, 1])."""
        idx = self.arm_to_index[arm]
        perf = self.performance[arm]
        
        # Update performance tracking
        perf.usage_count += 1
        perf.total_reward += reward
        
        # Convert reward to success/failure for Beta update
        # Reward > 0.5 is considered success
        if reward > 0.5:
            perf.successes += 1
            self.alpha[idx] += 1
        else:
            perf.failures += 1
            self.beta[idx] += 1
    
    def get_best_arm(self) -> Tuple[str, float]:
        """Get the arm with highest expected reward."""
        best_arm = max(self.arms, key=lambda arm: self.performance[arm].average_reward)
        return best_arm, self.performance[best_arm].average_reward
    
    def get_confidence_intervals(self) -> Dict[str, Tuple[float, float]]:
        """Get 95% confidence intervals for each arm."""
        intervals = {}
        for i, arm in enumerate(self.arms):
            # 95% confidence interval for Beta distribution
            samples = np.random.beta(self.alpha[i], self.beta[i], 10000)
            intervals[arm] = (np.percentile(samples, 2.5), np.percentile(samples, 97.5))
        return intervals


class AdaptiveRestartManager:
    """Manages adaptive restart strategy for hierarchical bandits."""
    
    def __init__(self, 
                 window_size: int = 20,
                 stagnation_threshold: float = 0.05,
                 min_performance_threshold: float = 0.3,
                 restart_cooldown: int = 50):
        
        self.window_size = window_size
        self.stagnation_threshold = stagnation_threshold
        self.min_performance_threshold = min_performance_threshold
        self.restart_cooldown = restart_cooldown
        
        # Performance tracking
        self.reward_history = deque(maxlen=window_size * 2)
        self.last_restart_step = 0
        self.step_count = 0
        
        # Restart strategies
        self.restart_strategies = ['reset_worst', 'diversity_boost', 'exploration_boost']
        self.restart_performance = defaultdict(list)
        
    def should_restart(self, recent_rewards: List[float]) -> bool:
        """Determine if restart is needed."""
        self.step_count += 1
        self.reward_history.extend(recent_rewards)
        
        # Don't restart too frequently
        if self.step_count - self.last_restart_step < self.restart_cooldown:
            return False
        
        if len(self.reward_history) < self.window_size:
            return False
        
        # Check for stagnation
        recent_window = list(self.reward_history)[-self.window_size:]
        older_window = list(self.reward_history)[-2*self.window_size:-self.window_size]
        
        if len(older_window) < self.window_size:
            return False
        
        recent_avg = np.mean(recent_window)
        older_avg = np.mean(older_window)
        improvement = recent_avg - older_avg
        
        # Restart conditions
        stagnation = abs(improvement) < self.stagnation_threshold
        poor_performance = recent_avg < self.min_performance_threshold
        
        return stagnation or poor_performance
    
    def execute_restart(self, bandits: Dict[str, ThompsonSamplingBandit]) -> str:
        """Execute restart strategy."""
        self.last_restart_step = self.step_count
        
        # Choose restart strategy based on past performance
        strategy = self._select_restart_strategy()
        
        if strategy == 'reset_worst':
            self._reset_worst_components(bandits)
        elif strategy == 'diversity_boost':
            self._boost_diversity(bandits)
        else:  # exploration_boost
            self._boost_exploration(bandits)
        
        return strategy
    
    def _select_restart_strategy(self) -> str:
        """Select restart strategy based on past performance."""
        if not self.restart_performance:
            return np.random.choice(self.restart_strategies)
        
        # Choose strategy with best average performance
        strategy_scores = {}
        for strategy in self.restart_strategies:
            if strategy in self.restart_performance:
                strategy_scores[strategy] = np.mean(self.restart_performance[strategy])
            else:
                strategy_scores[strategy] = 0.0
        
        return max(strategy_scores, key=strategy_scores.get)
    
    def _reset_worst_components(self, bandits: Dict[str, ThompsonSamplingBandit]):
        """Reset parameters for worst performing components."""
        for component_name, bandit in bandits.items():
            # Find worst performing arm
            worst_arm = min(bandit.arms, key=lambda arm: bandit.performance[arm].average_reward)
            worst_idx = bandit.arm_to_index[worst_arm]
            
            # Reset to prior
            bandit.alpha[worst_idx] = 1.0
            bandit.beta[worst_idx] = 1.0
            bandit.performance[worst_arm] = ComponentPerformance()
    
    def _boost_diversity(self, bandits: Dict[str, ThompsonSamplingBandit]):
        """Boost exploration by reducing confidence in all arms."""
        for bandit in bandits.values():
            # Reduce confidence by scaling down parameters
            bandit.alpha *= 0.7
            bandit.beta *= 0.7
            bandit.alpha = np.maximum(bandit.alpha, 1.0)  # Don't go below prior
            bandit.beta = np.maximum(bandit.beta, 1.0)
    
    def _boost_exploration(self, bandits: Dict[str, ThompsonSamplingBandit]):
        """Boost exploration for underexplored arms."""
        for bandit in bandits.values():
            # Find arms with low usage
            usage_counts = [bandit.performance[arm].usage_count for arm in bandit.arms]
            min_usage = min(usage_counts)
            
            for i, arm in enumerate(bandit.arms):
                if bandit.performance[arm].usage_count <= min_usage + 2:
                    # Boost exploration for underused arms
                    bandit.alpha[i] += 0.5
                    bandit.beta[i] += 0.5


class HierarchicalBanditAgent:
    """Main hierarchical bandit agent for strategy optimization."""
    
    def __init__(self):
        # Component-level bandits
        self.tone_bandit = ThompsonSamplingBandit(TONES)
        self.topic_bandit = ThompsonSamplingBandit(TOPICS)
        self.emotion_bandit = ThompsonSamplingBandit(EMOTIONS)
        self.hook_bandit = ThompsonSamplingBandit(HOOKS)
        
        self.bandits = {
            'tone': self.tone_bandit,
            'topic': self.topic_bandit, 
            'emotion': self.emotion_bandit,
            'hook': self.hook_bandit
        }
        
        # Meta-bandit for component importance
        self.meta_bandit = ThompsonSamplingBandit(['tone', 'topic', 'emotion', 'hook'])
        
        # Adaptive restart manager
        self.restart_manager = AdaptiveRestartManager()
        
        # Strategy tracking
        self.strategy_history = []
        self.recent_rewards = deque(maxlen=20)
        self.step_count = 0
        
    def select_strategy(self) -> Strategy:
        """Select strategy using hierarchical bandits."""
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
        
        return strategy
    
    def update(self, strategy: Strategy, reward: float):
        """Update bandits with strategy performance."""
        # Normalize reward to [0, 1] range
        normalized_reward = max(0, min(1, (reward + 1) / 2))  # Assumes reward in [-1, 1]
        
        # Update component bandits
        self.tone_bandit.update(strategy.tone, normalized_reward)
        self.topic_bandit.update(strategy.topic, normalized_reward)
        self.emotion_bandit.update(strategy.emotion, normalized_reward)
        self.hook_bandit.update(strategy.hook, normalized_reward)
        
        # Track recent performance
        self.recent_rewards.append(normalized_reward)
        
        # Check for restart
        if len(self.recent_rewards) >= 5 and self.restart_manager.should_restart([normalized_reward]):
            restart_strategy = self.restart_manager.execute_restart(self.bandits)
            print(f"\ud83d\udd04 Adaptive restart executed: {restart_strategy}")
            
            # Track restart performance
            window_before = list(self.recent_rewards)[-10:-5] if len(self.recent_rewards) >= 10 else []
            if window_before:
                self.restart_manager.restart_performance[restart_strategy].extend(window_before)
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        summary = {
            'step_count': self.step_count,
            'recent_performance': list(self.recent_rewards)[-10:],
            'average_recent_reward': np.mean(list(self.recent_rewards)[-10:]) if self.recent_rewards else 0,
            'components': {}
        }
        
        # Component performance
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
                'performance': {
                    arm: {
                        'successes': perf.successes,
                        'failures': perf.failures,
                        'total_reward': perf.total_reward,
                        'usage_count': perf.usage_count
                    }
                    for arm, perf in bandit.performance.items()
                }
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
        
        # Load bandit states
        for name, bandit_state in state.get('bandits', {}).items():
            if name in self.bandits:
                bandit = self.bandits[name]
                bandit.alpha = np.array(bandit_state['alpha'])
                bandit.beta = np.array(bandit_state['beta'])
                
                for arm, perf_data in bandit_state['performance'].items():
                    if arm in bandit.performance:
                        perf = bandit.performance[arm]
                        perf.successes = perf_data['successes']
                        perf.failures = perf_data['failures']
                        perf.total_reward = perf_data['total_reward']
                        perf.usage_count = perf_data['usage_count']
        
        # Load restart manager state
        restart_state = state.get('restart_manager', {})
        self.restart_manager.reward_history = deque(
            restart_state.get('reward_history', []), 
            maxlen=self.restart_manager.window_size * 2
        )
        self.restart_manager.last_restart_step = restart_state.get('last_restart_step', 0)
        self.restart_manager.step_count = restart_state.get('step_count', 0)
        self.restart_manager.restart_performance = defaultdict(
            list, restart_state.get('restart_performance', {})
        )

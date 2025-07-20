import numpy as np
from collections import deque, defaultdict
from typing import List, Tuple, Optional, Dict

from rl.strategy import Strategy, StrategySpace
from .embedding_service import EmbeddingService


class ConversationContext:
    """Store recent conversation turns for context embeddings."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.user_messages = deque(maxlen=window_size)
        self.ai_responses = deque(maxlen=window_size)
        self.strategies_used = deque(maxlen=window_size)
        self.engagement_scores = deque(maxlen=window_size)

    def add_turn(self, user_msg: str, ai_response: str, strategy: Strategy, engagement: float):
        self.user_messages.append(user_msg or "[Silent]")
        self.ai_responses.append(ai_response)
        self.strategies_used.append(strategy)
        self.engagement_scores.append(engagement)

    def get_recent_context(self, n: int = 3) -> Tuple[List[str], List[str], List[Strategy], List[float]]:
        return (
            list(self.user_messages)[-n:],
            list(self.ai_responses)[-n:],
            list(self.strategies_used)[-n:],
            list(self.engagement_scores)[-n:],
        )


class ContextualBanditAgent:
    """Contextual bandit using simple contextual UCB."""

    def __init__(self, context_window_size: int = 5):
        self.context = ConversationContext(context_window_size)
        self.embedding_service = EmbeddingService()
        self.strategy_space = StrategySpace()

        # Experience storage
        self.strategy_rewards: Dict[str, List[float]] = defaultdict(list)
        self.strategy_contexts: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.total_selections = 0

    # ------------------------------------------------------------------
    # Context vector construction
    # ------------------------------------------------------------------
    def _build_context_vector(self) -> np.ndarray:
        vectors: List[np.ndarray] = []
        user_msgs, ai_responses, strategies, engagements = self.context.get_recent_context(3)

        if user_msgs:
            user_text = " [TURN] ".join(user_msgs)
            vectors.append(self.embedding_service.embed_text(user_text))

        if ai_responses:
            ai_text = " [TURN] ".join(ai_responses)
            vectors.append(self.embedding_service.embed_text(ai_text))

        if strategies:
            strategy_embeddings = [self.embedding_service.embed_strategy(s) for s in strategies]
            vectors.append(np.mean(strategy_embeddings, axis=0))

        if engagements:
            vectors.append(self.embedding_service.create_engagement_features(engagements))

        if vectors:
            return np.concatenate(vectors)
        return np.zeros(1536 * 3 + 6)

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------
    def select_strategy(self) -> Strategy:
        self.total_selections += 1
        context_vector = self._build_context_vector()
        candidates = self._generate_candidates()
        best_strategy = None
        best_score = float('-inf')
        for candidate in candidates:
            score = self._calculate_contextual_ucb(candidate, context_vector)
            if score > best_score:
                best_score = score
                best_strategy = candidate
        return best_strategy or self.strategy_space.get_random_strategy()

    def _generate_candidates(self) -> List[Strategy]:
        candidates = []
        candidates.extend(self._get_top_performing_strategies(5))
        candidates.extend(self._get_contextually_similar_strategies(5))
        for _ in range(10):
            candidates.append(self.strategy_space.get_random_strategy())
        return candidates

    def _get_top_performing_strategies(self, n: int) -> List[Strategy]:
        strategy_avgs = []
        for key, rewards in self.strategy_rewards.items():
            if len(rewards) >= 3:
                strategy_avgs.append((key, np.mean(rewards)))
        strategy_avgs.sort(key=lambda x: x[1], reverse=True)
        top = []
        for key, _ in strategy_avgs[:n]:
            top.append(Strategy.from_key(key))
        return top

    def _get_contextually_similar_strategies(self, n: int) -> List[Strategy]:
        # Placeholder: currently returns empty list. Could be improved with embeddings.
        return []

    def _calculate_contextual_ucb(self, strategy: Strategy, context_vector: np.ndarray) -> float:
        key = strategy.to_key()
        if key not in self.strategy_rewards or not self.strategy_rewards[key]:
            return 10.0
        rewards = self.strategy_rewards[key]
        contexts = self.strategy_contexts[key]
        predicted_reward = self._predict_contextual_reward(strategy, context_vector, contexts, rewards)
        n = len(rewards)
        confidence_bonus = 2.0 * np.sqrt(np.log(self.total_selections) / n)
        return predicted_reward + confidence_bonus

    def _predict_contextual_reward(self, strategy: Strategy, current_context: np.ndarray,
                                   historical_contexts: List[np.ndarray], historical_rewards: List[float]) -> float:
        if not historical_contexts:
            return 0.5
        similarities = []
        for ctx in historical_contexts:
            sim = np.dot(current_context, ctx) / (np.linalg.norm(current_context) * np.linalg.norm(ctx) + 1e-8)
            similarities.append(sim)
        similarities = np.array(similarities)
        rewards = np.array(historical_rewards)
        k = min(5, len(similarities))
        top_idx = np.argsort(similarities)[-k:]
        weights = similarities[top_idx]
        weights = weights / (np.sum(weights) + 1e-8)
        return float(np.sum(weights * rewards[top_idx]))

    # ------------------------------------------------------------------
    # Updating
    # ------------------------------------------------------------------
    def update(self, strategy: Strategy, context_vector: np.ndarray, reward: float):
        key = strategy.to_key()
        self.strategy_rewards[key].append(reward)
        self.strategy_contexts[key].append(context_vector.copy())
        max_memory = 100
        if len(self.strategy_rewards[key]) > max_memory:
            self.strategy_rewards[key] = self.strategy_rewards[key][-max_memory:]
            self.strategy_contexts[key] = self.strategy_contexts[key][-max_memory:]

    # ------------------------------------------------------------------
    # Performance summary utilities
    # ------------------------------------------------------------------
    def _calculate_context_utilization(self) -> int:
        return len(self.context.user_messages)

    def _get_recent_performance(self) -> List[float]:
        recents = []
        for rewards in self.strategy_rewards.values():
            recents.extend(rewards[-5:])
        return recents[-20:]

    def _calculate_component_stats(self) -> Dict[str, Dict]:
        """Aggregate simple statistics for strategy components."""
        components = {
            'tone': {},
            'topic': {},
            'emotion': {},
            'hook': {},
        }

        for key, rewards in self.strategy_rewards.items():
            tone, topic, emotion, hook = key.split('|')
            for comp_name, arm in [
                ('tone', tone),
                ('topic', topic),
                ('emotion', emotion),
                ('hook', hook),
            ]:
                entry = components[comp_name].setdefault(arm, [])
                entry.extend(rewards)

        component_summary: Dict[str, Dict] = {}
        for comp_name, arms in components.items():
            if not arms:
                continue
            usage_stats = {}
            best_choice = None
            best_score = float('-inf')
            for arm, arm_rewards in arms.items():
                usage_count = len(arm_rewards)
                avg_reward = float(np.mean(arm_rewards)) if arm_rewards else 0.0
                recent_avg = float(np.mean(arm_rewards[-5:])) if arm_rewards else 0.0
                success_rate = float(sum(r > 0 for r in arm_rewards) / usage_count) if usage_count else 0.0
                usage_stats[arm] = {
                    'usage_count': usage_count,
                    'average_reward': avg_reward,
                    'recent_average': recent_avg,
                    'success_rate': success_rate,
                }
                if avg_reward > best_score:
                    best_score = avg_reward
                    best_choice = arm

            component_summary[comp_name] = {
                'best_choice': best_choice,
                'best_score': best_score,
                'confidence_intervals': {},
                'usage_stats': usage_stats,
            }

        return component_summary

    def get_performance_summary(self) -> Dict:
        """Return performance metrics compatible with old interface."""
        recent = self._get_recent_performance()
        summary = {
            'total_selections': self.total_selections,
            'strategies_tried': len(self.strategy_rewards),
            'top_strategies': [s.to_key() for s in self._get_top_performing_strategies(5)],
            'context_utilization': self._calculate_context_utilization(),
            'recent_rewards': recent,
            'average_recent_reward': float(np.mean(recent)) if recent else 0.0,
            'components': self._calculate_component_stats(),
            'restart_stats': {
                'total_restarts': 0,
                'last_restart_step': 0,
            },
        }
        return summary

    # Optional save/load for persistence
    def save(self, filepath):
        import pickle
        state = {
            'strategy_rewards': dict(self.strategy_rewards),
            'strategy_contexts': {k: np.stack(v).tolist() for k, v in self.strategy_contexts.items()},
            'total_selections': self.total_selections,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load(self, filepath):
        import pickle
        from pathlib import Path
        path = Path(filepath)
        if not path.exists():
            return
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.total_selections = state.get('total_selections', 0)
        for k, rewards in state.get('strategy_rewards', {}).items():
            self.strategy_rewards[k] = list(rewards)
        for k, ctxs in state.get('strategy_contexts', {}).items():
            self.strategy_contexts[k] = [np.array(c) for c in ctxs]


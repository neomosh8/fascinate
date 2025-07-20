import numpy as np
import random
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

    def _classify_context(self, user_msg: str, user_spoke: bool) -> str:
        """Classify the current context type for strategy selection."""
        if not user_spoke:
            return "auto_advance"
        elif not user_msg.strip():
            return "silent_user"
        elif self.total_selections <= 3:
            return "cold_start"
        elif len(user_msg.split()) <= 3:
            return "short_response"
        elif any(word in user_msg.lower() for word in ['?', 'what', 'how', 'why']):
            return "question"
        else:
            return "normal"

    def classify_current_context(self) -> str:
        recent_user_msgs, _, _, _ = self.context.get_recent_context(1)
        user_msg = recent_user_msgs[-1] if recent_user_msgs else ""
        user_spoke = len(user_msg.strip()) > 0 and user_msg != "[Silent]"
        return self._classify_context(user_msg, user_spoke)

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
        # Very early in the conversation - provide a non-zero vector
        start_vec = self.embedding_service.embed_text("conversation start")
        padding = np.zeros(1536 * 2 + 6)
        return np.concatenate([start_vec, padding])

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------
    def select_strategy(self) -> Strategy:
        self.total_selections += 1
        context_vector = self._build_context_vector()

        # Determine context type for smarter candidate generation
        recent_user_msgs, _, _, _ = self.context.get_recent_context(1)
        user_msg = recent_user_msgs[-1] if recent_user_msgs else ""
        user_spoke = len(user_msg.strip()) > 0 and user_msg != "[Silent]"
        context_type = self._classify_context(user_msg, user_spoke)

        candidates = self._generate_context_aware_candidates(context_type)
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

    def _generate_context_aware_candidates(self, context_type: str) -> List[Strategy]:
        """Generate candidate strategies based on the classified context."""
        candidates: List[Strategy] = []

        if context_type == "auto_advance":
            candidates.extend(self._get_continuation_strategies())
            engaging_hooks = ["you know what?", "are you with me?", "listen"]
            for hook in engaging_hooks:
                candidates.append(
                    Strategy(
                        tone=random.choice(["playful", "informational"]),
                        topic=random.choice(["facts", "story"]),
                        emotion=random.choice(["happy", "serious"]),
                        hook=hook,
                        index=self.total_selections,
                    )
                )
        elif context_type == "cold_start":
            candidates.extend(self._get_safe_starter_strategies())

        # Base candidates from historical performance and randomness
        candidates.extend(self._generate_candidates())
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

    def _get_continuation_strategies(self) -> List[Strategy]:
        """Return strategies that continue from the previous turn."""
        if not self.context.strategies_used:
            return []
        last_strategy = self.context.strategies_used[-1]
        return [last_strategy]

    def _get_safe_starter_strategies(self) -> List[Strategy]:
        """Return a set of safe starter strategies for cold start."""
        starters = [
            Strategy(tone="calm", topic="facts", emotion="happy", hook="hey [name]", index=0),
            Strategy(tone="kind", topic="story", emotion="serious", hook="hey [name]", index=0),
        ]
        return starters

    def _calculate_contextual_ucb(self, strategy: Strategy, context_vector: np.ndarray) -> float:
        key = strategy.to_key()
        if key not in self.strategy_rewards or not self.strategy_rewards[key]:
            return self._calculate_cold_start_score(strategy, context_vector)
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

    def _calculate_cold_start_score(self, strategy: Strategy, context_vector: np.ndarray) -> float:
        """Estimate score for unseen strategies using component priors."""
        base_score = 8.0
        component_bonuses = {
            'tone': {'calm': 1.5, 'kind': 1.2, 'informational': 1.0, 'playful': 0.8},
            'topic': {'facts': 1.2, 'story': 1.0, 'nerds': 0.8},
            'emotion': {'happy': 1.2, 'serious': 1.0, 'whisper': 0.8},
            'hook': {'hey [name]': 1.0, 'you know what?': 0.9},
        }

        for comp, value in [
            ('tone', strategy.tone),
            ('topic', strategy.topic),
            ('emotion', strategy.emotion),
            ('hook', strategy.hook),
        ]:
            bonus = component_bonuses.get(comp, {}).get(value, 0.5)
            base_score += bonus

        if self.total_selections <= 3:
            if strategy.tone in ['calm', 'kind'] and strategy.emotion in ['happy', 'serious']:
                base_score += 2.0

        return base_score

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


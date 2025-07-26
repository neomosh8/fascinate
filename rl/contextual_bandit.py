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
        if self.total_selections == 0:
            return "cold_start"
        if not user_spoke:
            return "auto_advance"
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
        """Build context vector with normalized components to prevent scale dominance."""
        vectors: List[np.ndarray] = []
        user_msgs, ai_responses, strategies, engagements = self.context.get_recent_context(3)

        # 1. User text embeddings (normalized)
        if user_msgs:
            user_text = " [TURN] ".join(user_msgs)
            user_embed = self.embedding_service.embed_text(user_text)
            # L2 normalize to unit vector
            user_norm = np.linalg.norm(user_embed)
            if user_norm > 1e-8:  # Avoid division by zero
                user_embed = user_embed / user_norm
            vectors.append(user_embed)
        else:
            # Zero vector if no user messages
            vectors.append(np.zeros(1536))  # OpenAI embedding dimension

        # 2. AI response embeddings (normalized)
        if ai_responses:
            ai_text = " [TURN] ".join(ai_responses)
            ai_embed = self.embedding_service.embed_text(ai_text)
            # L2 normalize to unit vector
            ai_norm = np.linalg.norm(ai_embed)
            if ai_norm > 1e-8:
                ai_embed = ai_embed / ai_norm
            vectors.append(ai_embed)
        else:
            # Zero vector if no AI responses
            vectors.append(np.zeros(1536))

        # 3. Strategy embeddings (sequence-aware with position weighting)
        if strategies:
            # Position weights: recent strategies get higher influence
            pos_weights = [1.0, 0.8, 0.6]  # Most recent first
            strategy_sequence = []

            for i, strategy in enumerate(strategies):
                strategy_embed = self.embedding_service.embed_strategy(strategy)
                # L2 normalize individual strategy
                strategy_norm = np.linalg.norm(strategy_embed)
                if strategy_norm > 1e-8:
                    strategy_embed = strategy_embed / strategy_norm

                # Apply position weight
                weight = pos_weights[i] if i < len(pos_weights) else 0.4
                strategy_embed = strategy_embed * weight
                strategy_sequence.append(strategy_embed)

            # Pad with zeros if we have fewer than 3 strategies
            while len(strategy_sequence) < 3:
                strategy_sequence.append(np.zeros(1536))

            # Concatenate sequence (preserves order)
            strategy_concat = np.concatenate(strategy_sequence)
            vectors.append(strategy_concat)
        else:
            # Zero vector if no strategies (3 strategy positions)
            vectors.append(np.zeros(1536 * 3))

        # 4. Engagement features (normalized)
        if engagements:
            eng_features = self.embedding_service.create_engagement_features(engagements)
            # L2 normalize engagement features
            eng_norm = np.linalg.norm(eng_features)
            if eng_norm > 1e-8:
                eng_features = eng_features / eng_norm
            vectors.append(eng_features)
        else:
            # Zero vector if no engagement data
            vectors.append(np.zeros(6))  # 6 engagement features

        # 5. Concatenate all normalized components
        if vectors:
            context_vector = np.concatenate(vectors)
            # Optional: normalize the final concatenated vector as well
            final_norm = np.linalg.norm(context_vector)
            if final_norm > 1e-8:
                context_vector = context_vector / final_norm
            return context_vector

        # Fallback for very early conversation
        start_vec = self.embedding_service.embed_text("conversation start")
        start_norm = np.linalg.norm(start_vec)
        if start_norm > 1e-8:
            start_vec = start_vec / start_norm

        # Create normalized zero padding for other components
        padding = np.zeros(1536 * 2 + 6)  # AI text + strategy + engagement
        fallback_vector = np.concatenate([start_vec, padding])

        # Normalize the fallback vector too
        fallback_norm = np.linalg.norm(fallback_vector)
        if fallback_norm > 1e-8:
            fallback_vector = fallback_vector / fallback_norm

        return fallback_vector

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

        if context_type == "cold_start":
            candidates = self._get_safe_starter_strategies()
        elif context_type == "auto_advance":
            candidates = self._get_continuation_strategies() + [
                Strategy(
                    tone=random.choice(["playful", "informational"]),
                    topic=random.choice(["facts", "story"]),
                    emotion=random.choice(["happy", "serious"]),
                    hook=hook,
                    index=self.total_selections,
                )
                for hook in ["you know what?", "are you with me?", "listen"]
            ]
        else:
            candidates = self._get_top_performing_strategies(5)
            if not candidates:
                candidates.append(self.strategy_space.get_random_strategy())

        best_strategy = max(
            candidates,
            key=lambda cand: self._calculate_contextual_ucb(cand, context_vector),
        )
        return best_strategy


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

    def _calculate_contextual_similarity(self, current_context: np.ndarray, historical_context: np.ndarray) -> float:
        """
        Calculate similarity using separate text and engagement similarity computation.
        Preserves OpenAI embedding semantics while handling scale/dimensionality issues.
        """
        # Split contexts into components
        # Assuming structure: [user_embed(1536) + ai_embed(1536) + strategy_embed(1536) + engagement(6)]
        TEXT_DIM = 7680  # 3 * 1536

        curr_text = current_context[:TEXT_DIM]
        curr_eng = current_context[TEXT_DIM:]

        hist_text = historical_context[:TEXT_DIM]
        hist_eng = historical_context[TEXT_DIM:]

        # Calculate text similarity (preserve OpenAI embedding space)
        text_norm_curr = np.linalg.norm(curr_text)
        text_norm_hist = np.linalg.norm(hist_text)

        if text_norm_curr > 1e-8 and text_norm_hist > 1e-8:
            text_sim = np.dot(curr_text, hist_text) / (text_norm_curr * text_norm_hist)
        else:
            text_sim = 0.0

        # Calculate engagement similarity
        eng_norm_curr = np.linalg.norm(curr_eng)
        eng_norm_hist = np.linalg.norm(hist_eng)

        if eng_norm_curr > 1e-8 and eng_norm_hist > 1e-8:
            eng_sim = np.dot(curr_eng, hist_eng) / (eng_norm_curr * eng_norm_hist)
        else:
            eng_sim = 0.0

        # Combine similarities with weights
        # You can tune these weights based on what matters more for your use case
        text_weight = 0.5  # Text context importance
        engagement_weight = 0.5  # Engagement pattern importance

        final_similarity = text_weight * text_sim + engagement_weight * eng_sim
        print(text_sim,eng_sim)
        # Clip to valid cosine similarity range
        return np.clip(final_similarity, -1.0, 1.0)

    def _predict_contextual_reward(self, strategy: Strategy, current_context: np.ndarray,
                                   historical_contexts: List[np.ndarray], historical_rewards: List[float]) -> float:
        """
        Predict reward using separate text/engagement similarity calculation.
        """
        if not historical_contexts:
            return 0.5

        # Calculate similarities using the new method
        similarities = []
        for hist_context in historical_contexts:
            sim = self._calculate_contextual_similarity(current_context, hist_context)
            similarities.append(sim)

        similarities = np.array(similarities)
        rewards = np.array(historical_rewards)

        # Find top-k most similar contexts
        k = min(5, len(similarities))
        top_indices = np.argsort(similarities)[-k:]

        # Weight rewards by similarity
        top_similarities = similarities[top_indices]
        top_rewards = rewards[top_indices]

        # Avoid division by zero
        similarity_sum = np.sum(top_similarities)
        if similarity_sum > 1e-8:
            weights = top_similarities / similarity_sum
            predicted_reward = float(np.sum(weights * top_rewards))
        else:
            # Fallback to simple average if no meaningful similarities
            predicted_reward = float(np.mean(top_rewards))

        return predicted_reward
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


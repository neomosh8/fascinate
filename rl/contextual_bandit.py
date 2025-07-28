import numpy as np
from collections import deque, defaultdict
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


from rl.strategy import Strategy, StrategySpace
from .embedding_service import EmbeddingService


class ConversationContext:
    """Store recent conversation turns for context embeddings."""

    def __init__(self, window_size: int = 10):
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


@dataclass
class TurnContext:
    """Structured context for a single decision."""

    session_phase: str
    target_concept_embedding: Optional[np.ndarray]
    user_embedding: Optional[np.ndarray]
    ai_embedding: Optional[np.ndarray]
    strategy_sequence_embedding: Optional[np.ndarray]
    engagement_features: Optional[np.ndarray]


class ContextualBanditAgent:
    """Contextual bandit using simple contextual UCB."""

    def __init__(self, context_window_size: int = 5):
        self.context = ConversationContext(context_window_size)
        self.embedding_service = EmbeddingService()
        self.strategy_space = StrategySpace()

        # Experience storage
        self.strategy_rewards: Dict[str, List[float]] = defaultdict(list)
        self.strategy_contexts: Dict[str, List[TurnContext]] = defaultdict(list)
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
    def _build_context_vector(
        self,
        session_phase: str,
        target_concept: Optional[str],
    ) -> TurnContext:
        """Build a structured context object with raw embeddings."""
        user_msgs, ai_responses, strategies, engagements = self.context.get_recent_context(3)

        target_concept_embedding = (
            self.embedding_service.embed_text(target_concept) if target_concept else None
        )

        user_embedding = (
            self.embedding_service.embed_text(" [TURN] ".join(user_msgs)) if user_msgs else None
        )

        ai_embedding = (
            self.embedding_service.embed_text(" [TURN] ".join(ai_responses)) if ai_responses else None
        )

        strategy_sequence_embedding = None
        if strategies:
            pos_weights = [1.0, 0.8, 0.6]
            weighted = []
            for i, strategy in enumerate(strategies):
                embed = self.embedding_service.embed_strategy(strategy)
                weight = pos_weights[i] if i < len(pos_weights) else 0.4
                weighted.append(embed * weight)
            strategy_sequence_embedding = np.mean(weighted, axis=0)

        engagement_features = (
            self.embedding_service.create_engagement_features(engagements) if engagements else None
        )

        return TurnContext(
            session_phase=session_phase,
            target_concept_embedding=target_concept_embedding,
            user_embedding=user_embedding,
            ai_embedding=ai_embedding,
            strategy_sequence_embedding=strategy_sequence_embedding,
            engagement_features=engagement_features,
        )

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------
    def select_strategy(
        self,
        session_phase: str,
        target_concept: Optional[str],
        num_candidates: int = 15,
    ) -> Strategy:
        """Select the best strategy based on provided therapeutic context."""
        context_object = self._build_context_vector(session_phase, target_concept)

        all_tried_strategies = [Strategy.from_key(k) for k in self.strategy_rewards.keys()]

        if not all_tried_strategies:
            return self.strategy_space.get_random_strategy()

        scored_candidates = []
        for strat in all_tried_strategies:
            key = strat.to_key()
            contexts = self.strategy_contexts.get(key, [])
            rewards = self.strategy_rewards.get(key, [])
            predicted = self._predict_contextual_reward(strat, context_object, contexts, rewards)
            scored_candidates.append((strat, predicted))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [s for s, _ in scored_candidates[:num_candidates]]

        import random
        if random.random() < 0.2:
            print("🤖 Bandit is exploring (random choice)...")
            if session_phase == 'exploitation' and top_candidates:
                return self.strategy_space.get_mutated_strategy(top_candidates[0])
            return self.strategy_space.get_random_strategy()

        if not top_candidates:
            return self.strategy_space.get_random_strategy()

        best_strategy = top_candidates[0]
        self.total_selections += 1
        print(f"🤖 Bandit chose: {best_strategy.to_key()} for phase '{session_phase}'")
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
        """Return a set of specific starter strategies."""
        starters = [
            Strategy(
                index=-1,
                tone="warm",
                topic="professional",  # Pick any topic from config
                emotion="talk about how this therapy session will go",
                hook="hey maya"  # Pick any hook from config
            ),
            Strategy(
                index=-1,
                tone="warm",
                topic="facts",
                emotion="talk about how this therapy session will go",
                hook="hello there, good to see you! emmm i mean not quite literally , but anyway"
            )
        ]
        return starters


    def _cosine_similarity(self, vec1: Optional[np.ndarray], vec2: Optional[np.ndarray]) -> float:
        """Safely compute cosine similarity, returning 0 for missing vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _calculate_contextual_similarity(self, current_context: TurnContext, historical_context: TurnContext) -> float:
        """Weighted similarity across structured context components."""

        weights = {
            "phase": 0.10,
            "concept": 0.25,
            "user_text": 0.20,
            "ai_text": 0.10,
            "strategy": 0.15,
            "engagement": 0.20,
        }

        phase_sim = 1.0 if current_context.session_phase == historical_context.session_phase else 0.0

        concept_sim = self._cosine_similarity(
            current_context.target_concept_embedding,
            historical_context.target_concept_embedding,
        )

        user_sim = self._cosine_similarity(current_context.user_embedding, historical_context.user_embedding)
        ai_sim = self._cosine_similarity(current_context.ai_embedding, historical_context.ai_embedding)
        strat_sim = self._cosine_similarity(
            current_context.strategy_sequence_embedding,
            historical_context.strategy_sequence_embedding,
        )

        curr_eng = current_context.engagement_features
        hist_eng = historical_context.engagement_features
        if curr_eng is not None:
            n = np.linalg.norm(curr_eng)
            if n > 1e-8:
                curr_eng = curr_eng / n
        if hist_eng is not None:
            n = np.linalg.norm(hist_eng)
            if n > 1e-8:
                hist_eng = hist_eng / n

        eng_sim = self._cosine_similarity(curr_eng, hist_eng)

        final_similarity = (
            weights["phase"] * phase_sim
            + weights["concept"] * concept_sim
            + weights["user_text"] * user_sim
            + weights["ai_text"] * ai_sim
            + weights["strategy"] * strat_sim
            + weights["engagement"] * eng_sim
        )

        return float(final_similarity)

    def _predict_contextual_reward(
        self,
        strategy: Strategy,
        current_context: TurnContext,
        historical_contexts: List[TurnContext],
        historical_rewards: List[float],
    ) -> float:
        """
        Predict reward using separate text/engagement similarity calculation.
        """
        if not historical_contexts:
            return 0.5

        # Calculate similarities using the new method
        similarities = [
            self._calculate_contextual_similarity(current_context, hist_ctx)
            for hist_ctx in historical_contexts
        ]
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

    # ------------------------------------------------------------------
    # Updating
    # ------------------------------------------------------------------
    def update(self, strategy: Strategy, context_object: TurnContext, reward: float):
        key = strategy.to_key()
        self.strategy_rewards[key].append(reward)
        self.strategy_contexts[key].append(context_object)
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

        def ctx_to_dict(ctx: TurnContext) -> Dict:
            return {
                'session_phase': ctx.session_phase,
                'target_concept_embedding': ctx.target_concept_embedding.tolist() if ctx.target_concept_embedding is not None else None,
                'user_embedding': ctx.user_embedding.tolist() if ctx.user_embedding is not None else None,
                'ai_embedding': ctx.ai_embedding.tolist() if ctx.ai_embedding is not None else None,
                'strategy_sequence_embedding': ctx.strategy_sequence_embedding.tolist() if ctx.strategy_sequence_embedding is not None else None,
                'engagement_features': ctx.engagement_features.tolist() if ctx.engagement_features is not None else None,
            }

        state = {
            'strategy_rewards': dict(self.strategy_rewards),
            'strategy_contexts': {k: [ctx_to_dict(c) for c in v] for k, v in self.strategy_contexts.items()},
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

        def dict_to_ctx(d: Dict) -> TurnContext:
            return TurnContext(
                session_phase=d.get('session_phase', 'exploration'),
                target_concept_embedding=np.array(d['target_concept_embedding']) if d.get('target_concept_embedding') is not None else None,
                user_embedding=np.array(d['user_embedding']) if d.get('user_embedding') is not None else None,
                ai_embedding=np.array(d['ai_embedding']) if d.get('ai_embedding') is not None else None,
                strategy_sequence_embedding=np.array(d['strategy_sequence_embedding']) if d.get('strategy_sequence_embedding') is not None else None,
                engagement_features=np.array(d['engagement_features']) if d.get('engagement_features') is not None else None,
            )

        for k, ctxs in state.get('strategy_contexts', {}).items():
            self.strategy_contexts[k] = [dict_to_ctx(c) for c in ctxs]


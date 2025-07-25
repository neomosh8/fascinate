"""Manage therapeutic exploration-exploitation sessions."""

import asyncio
from typing import Dict, Optional

from .concept_extractor import ConceptExtractor
from .concept_tracker import ConceptTracker
from rl.therapeutic_strategy import TherapeuticStrategy
from config import TherapyConfig


class TherapeuticSessionManager:
    """Manage therapeutic session flow."""

    def __init__(self):
        self.concept_extractor = ConceptExtractor()
        self.concept_tracker = ConceptTracker()
        self.config = TherapyConfig()

        self.session_phase = "exploration"
        self.turn_count_in_phase = 0
        self.current_exploitation_target: Optional[str] = None

    async def select_therapeutic_strategy(self) -> TherapeuticStrategy:
        """Select strategy based on current therapeutic phase."""
        if self.session_phase == "exploration":
            return TherapeuticStrategy.create_exploration_strategy(self.turn_count_in_phase)
        elif self.session_phase == "exploitation":
            if not self.current_exploitation_target:
                hot_concepts = self.concept_tracker.get_hot_concepts(top_k=1)
                if hot_concepts:
                    self.current_exploitation_target = hot_concepts[0][0]
                else:
                    self._switch_to_exploration()
                    return TherapeuticStrategy.create_exploration_strategy(0)
            return TherapeuticStrategy.create_exploitation_strategy(
                self.current_exploitation_target, self.turn_count_in_phase
            )

    async def process_therapeutic_turn(
        self, user_text: str, ai_response: str, engagement_after: float
    ) -> Dict:
        """Process turn and update therapeutic state."""
        all_text = f"{user_text} {ai_response}"
        concepts = await self.concept_extractor.extract_concepts(all_text)
        if concepts:
            self.concept_tracker.record_concept_activation(concepts, engagement_after)

        self._update_session_phase()

        return {
            "concepts_found": concepts,
            "session_phase": self.session_phase,
            "turn_in_phase": self.turn_count_in_phase,
            "hot_concepts": self.concept_tracker.get_hot_concepts(),
            "current_target": self.current_exploitation_target,
        }

    def _update_session_phase(self):
        self.turn_count_in_phase += 1
        if self.session_phase == "exploration":
            hot_concepts = self.concept_tracker.get_hot_concepts(
                min_mentions=self.config.min_concept_mentions
            )
            if hot_concepts and self.turn_count_in_phase >= self.config.exploration_turns:
                self._switch_to_exploitation(hot_concepts[0][0])
        elif self.session_phase == "exploitation":
            if self.turn_count_in_phase >= self.config.exploitation_turns:
                self._switch_to_exploration()

    def _switch_to_exploration(self):
        self.session_phase = "exploration"
        self.turn_count_in_phase = 0
        self.current_exploitation_target = None
        print("🔍 Switching to EXPLORATION mode")

    def _switch_to_exploitation(self, target_concept: str):
        self.session_phase = "exploitation"
        self.turn_count_in_phase = 0
        self.current_exploitation_target = target_concept
        print(f"🎯 Switching to EXPLOITATION mode - targeting: {target_concept}")

    def get_session_summary(self) -> Dict:
        """Get therapeutic session summary."""
        return {
            "session_phase": self.session_phase,
            "turn_in_phase": self.turn_count_in_phase,
            "exploration_status": self.concept_tracker.get_exploration_status(),
            "current_target": self.current_exploitation_target,
        }

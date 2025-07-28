"""Manage therapeutic exploration-exploitation sessions."""

import asyncio
from typing import Dict, Optional

from .concept_extractor import ConceptExtractor
from .concept_tracker import ConceptTracker
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


    async def process_therapeutic_turn(
            self, user_text: str, ai_response: str, engagement_after: float, emotion_after: float  # NEW parameter
    ) -> Dict:
        """Process turn and update therapeutic state."""
        all_text = f"{user_text} {ai_response}"
        concepts = await self.concept_extractor.extract_concepts(all_text)
        if concepts:
            self.concept_tracker.record_concept_activation(concepts, engagement_after, emotion_after)  # Updated

        self._update_session_phase()

        return {
            "concepts_found": concepts,
            "session_phase": self.session_phase,
            "turn_in_phase": self.turn_count_in_phase,
            "hot_concepts": self.concept_tracker.get_hot_concepts(),
            "current_target": self.current_exploitation_target,
            "emotional_profiles": {  # NEW: Add emotional context
                concept: self.concept_tracker.get_emotional_profile(concept)
                for concept, _ in self.concept_tracker.get_hot_concepts()
            }
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

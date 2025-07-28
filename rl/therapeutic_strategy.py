"""Therapeutic strategy with voice parameters."""

# Import the EmbeddingService
from rl.embedding_service import EmbeddingService

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import random

from rl.strategy import Strategy
from config import (
    THERAPEUTIC_TONES,
    EXPLORATION_DOMAINS,
    THERAPEUTIC_APPROACHES,
    THERAPEUTIC_HOOKS,
    THERAPEUTIC_MODALITIES
)


# --- NEW: Helper function for comparing embeddings ---
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # Avoid division by zero
    return dot_product / (norm_vec1 * norm_vec2)


# --- NEW: Function to select the best approach using embeddings ---
def choose_best_approach_by_embedding(target_concept: str, embedding_service: EmbeddingService) -> str:
    """
    Chooses the most relevant therapeutic approach by comparing embeddings.
    This is more semantically robust than keyword matching.
    """
    if not target_concept.strip():
        return random.choice(THERAPEUTIC_APPROACHES)

    concept_embedding = embedding_service.embed_text(target_concept)

    similarities = {}
    for approach, data in THERAPEUTIC_MODALITIES.items():
        # Embed the rich description of the modality
        approach_embedding = embedding_service.embed_text(data['description'])
        # Calculate and store the similarity score
        sim = cosine_similarity(concept_embedding, approach_embedding)
        similarities[approach] = sim

    # Return the approach with the highest similarity score
    if not similarities:
        return random.choice(THERAPEUTIC_APPROACHES)

    return max(similarities, key=similarities.get)


@dataclass
class TherapeuticStrategy(Strategy):
    """Strategy with therapeutic focus and learnable TTS voice parameters."""

    exploration_mode: bool = True
    target_concept: Optional[str] = None

    base_voice_speed: float = 1.0
    base_voice_pitch: float = 0.0
    base_voice_energy: float = 1.0
    voice_warmth: float = 1.0

    def __init__(
        self,
        tone: str,
        domain: str,
        approach: str,
        hook: str,
        exploration_mode: bool = True,
        target_concept: Optional[str] = None,
        index: int = 0,
    ):
        # The superclass maps these to self.tone, self.topic, self.emotion, self.hook
        super().__init__(tone, domain, approach, hook, index)
        self.exploration_mode = exploration_mode
        self.target_concept = target_concept
        self._initialize_voice_parameters()

    def _initialize_voice_parameters(self):
        tone_voice_map = {
            "empathetic": {"speed": 0.9, "pitch": 0.1, "energy": 0.9, "warmth": 1.2},
            "validating": {"speed": 0.85, "pitch": 0.0, "energy": 0.8, "warmth": 1.1},
            "curious": {"speed": 1.0, "pitch": 0.05, "energy": 1.1, "warmth": 1.0},
            "gentle": {"speed": 0.8, "pitch": 0.0, "energy": 0.7, "warmth": 1.2},
            "reflective": {"speed": 0.85, "pitch": -0.05, "energy": 0.8, "warmth": 1.0},
            "supportive": {"speed": 0.9, "pitch": 0.1, "energy": 0.9, "warmth": 1.3},
        }

        if self.tone in tone_voice_map:
            params = tone_voice_map[self.tone]
            self.base_voice_speed = params["speed"]
            self.base_voice_pitch = params["pitch"]
            self.base_voice_energy = params["energy"]
            self.voice_warmth = params["warmth"]

    def to_prompt(self) -> str:
        # --- MODIFIED: Use the rich description from the dictionary ---
        # self.emotion holds the approach name (e.g., "ifs") from the __init__ call
        approach_description = THERAPEUTIC_MODALITIES[self.emotion]['description']

        if self.exploration_mode:
            # The prompt now includes the detailed instructions for the chosen approach
            return (
                f"Adopt a {self.tone} therapeutic tone. Gently explore topics related to {self.topic} "
                f"using {approach_description} " # Changed from `{self.emotion} approach`
                f"Use a natural variation of '{self.hook}' in talking and see what resonates. "
                "Keep it exploratory and light - you're discovering what matters to this person. "
                "Be curious about their inner world. Don't go too deep yet, just see what emerges."
            )
        else:
            return (
                f"Use a {self.tone} therapeutic tone to deeply explore the concept of '{self.target_concept}'. "
                f"To do this, apply {approach_description} " # Changed from `Apply {self.emotion} therapeutic techniques`
                f"Use a natural variation of '{self.hook}' in your talking to go deeper. "
                "This concept triggered significant emotional activation. Make the conversation meaningful "
                "and directed. Do not use bullet points or numbered lists."
            )

    def get_emotion_adapted_tts_params(self, user_emotion: float, user_engagement: float) -> Dict[str, float]:
        adapted_speed = self.base_voice_speed
        adapted_pitch = self.base_voice_pitch
        adapted_energy = self.base_voice_energy
        adapted_warmth = self.voice_warmth

        if user_emotion < 0.4:
            adapted_speed *= 0.9
            adapted_pitch -= 0.05
            adapted_energy *= 0.8
            adapted_warmth *= 1.2
        elif user_emotion > 0.6:
            adapted_speed *= 1.05
            adapted_energy *= 1.1
            adapted_warmth *= 1.1

        if user_engagement < 0.4:
            adapted_energy *= 1.15
            adapted_speed *= 1.05
        elif user_engagement > 0.7:
            adapted_energy *= 0.95

        return {
            "speed": np.clip(adapted_speed, 0.7, 1.3),
            "pitch": np.clip(adapted_pitch, -0.3, 0.3),
            "energy": np.clip(adapted_energy, 0.7, 1.3),
            "warmth": np.clip(adapted_warmth, 0.7, 1.3),
        }

    def evolve_voice_parameters(self, reward: float, learning_rate: float = 0.1):
        if abs(reward) < 0.1:
            return
        mutation_strength = learning_rate * reward
        if reward > 0:
            self.base_voice_speed += random.uniform(-0.02, 0.02) + mutation_strength * 0.01
            self.base_voice_pitch += random.uniform(-0.01, 0.01) + mutation_strength * 0.005
            self.base_voice_energy += random.uniform(-0.02, 0.02) + mutation_strength * 0.01
            self.voice_warmth += random.uniform(-0.02, 0.02) + mutation_strength * 0.01
        else:
            self.base_voice_speed += random.uniform(-0.05, 0.05)
            self.base_voice_pitch += random.uniform(-0.02, 0.02)
            self.base_voice_energy += random.uniform(-0.05, 0.05)
            self.voice_warmth += random.uniform(-0.05, 0.05)

        self.base_voice_speed = np.clip(self.base_voice_speed, 0.7, 1.3)
        self.base_voice_pitch = np.clip(self.base_voice_pitch, -0.3, 0.3)
        self.base_voice_energy = np.clip(self.base_voice_energy, 0.7, 1.3)
        self.voice_warmth = np.clip(self.voice_warmth, 0.7, 1.3)

    def get_voice_signature(self) -> str:
        return (
            f"spd:{self.base_voice_speed:.2f},pit:{self.base_voice_pitch:.2f}," 
            f"eng:{self.base_voice_energy:.2f},wrm:{self.voice_warmth:.2f}"
        )

    @classmethod
    def create_exploration_strategy(cls, index: int = 0) -> "TherapeuticStrategy":
        return cls(
            tone=random.choice(THERAPEUTIC_TONES),
            domain=random.choice(EXPLORATION_DOMAINS),
            approach=random.choice(THERAPEUTIC_APPROACHES),
            hook=random.choice(THERAPEUTIC_HOOKS),
            exploration_mode=True,
            index=index,
        )

    # --- MODIFIED: This method now requires an EmbeddingService instance ---
    @classmethod
    def create_exploitation_strategy(cls, target_concept: str, embedding_service: EmbeddingService, index: int = 0) -> "TherapeuticStrategy":
        # Intelligently choose the best approach using semantic similarity
        best_approach = choose_best_approach_by_embedding(target_concept, embedding_service)

        return cls(
            tone=random.choice(["empathetic", "validating", "supportive"]),
            domain="deep_exploration", # 'domain' maps to 'topic' in the superclass
            approach=best_approach,    # 'approach' maps to 'emotion' in the superclass
            hook=random.choice(THERAPEUTIC_HOOKS),
            exploration_mode=False,
            target_concept=target_concept,
            index=index,
        )
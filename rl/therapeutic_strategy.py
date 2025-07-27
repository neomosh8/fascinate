"""Therapeutic strategy with voice parameters."""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import random

from .strategy import Strategy
from config import (
    THERAPEUTIC_TONES,
    EXPLORATION_DOMAINS,
    THERAPEUTIC_APPROACHES,
    THERAPEUTIC_HOOKS,
)


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
        if self.exploration_mode:
            return (
                f"Adopt a {self.tone} therapeutic tone. Gently explore topics related to {self.topic} "
                f"using {self.emotion} approach. use a natural variation of  '{self.hook}' in talking and see what resonates. "
                "Keep it exploratory and light - you're discovering what matters to this person. "
                "Be curious about their inner world. Don't go too deep yet, just see what emerges."
            )
        else:
            return (
                f"Use a {self.tone} therapeutic tone to deeply explore {self.target_concept}. "
                f"Apply {self.emotion} therapeutic techniques. use a natural variation of  '{self.hook}' in your talking and go deeper "
                "into this area that showed strong emotional activation. This concept triggered significant "
                "neural response, make meaningful deep conversation turn by turn, don't hover and stall"
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
            f"spd:{self.base_voice_speed:.2f},pit:{self.base_voice_pitch:.2f}," \
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

    @classmethod
    def create_exploitation_strategy(cls, target_concept: str, index: int = 0) -> "TherapeuticStrategy":
        return cls(
            tone=random.choice(["empathetic", "validating", "supportive"]),
            domain="deep_exploration",
            approach=random.choice(["cognitive", "narrative", "somatic"]),
            hook=random.choice(THERAPEUTIC_HOOKS),
            exploration_mode=False,
            target_concept=target_concept,
            index=index,
        )

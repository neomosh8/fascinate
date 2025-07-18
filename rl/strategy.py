"""Communication strategy definitions and management."""

from dataclasses import dataclass
from typing import List, Tuple
import itertools
from config import TONES, TOPICS, EMOTIONS, HOOKS


@dataclass
class Strategy:
    """Represents a communication strategy."""
    tone: str
    topic: str
    emotion: str
    hook: str
    index: int

    def to_prompt(self) -> str:
        """Convert strategy to GPT system prompt."""
        return (
            f"Talk about {self.topic} with {self.tone} tone, "
            f"express {self.emotion} in your voice and use {self.hook} in your talking"
        )


class StrategySpace:
    """Manages the space of all possible strategies."""

    def __init__(self, subset_size: int = 100):
        # Generate all combinations
        all_combos = list(itertools.product(TONES, TOPICS, EMOTIONS, HOOKS))

        # For prototype, use a subset to keep manageable
        if len(all_combos) > subset_size:
            # Sample evenly across the space
            step = len(all_combos) // subset_size
            self.strategies = [
                Strategy(tone=c[0], topic=c[1], emotion=c[2], hook=c[3], index=i)
                for i, c in enumerate(all_combos[::step][:subset_size])
            ]
        else:
            self.strategies = [
                Strategy(tone=c[0], topic=c[1], emotion=c[2], hook=c[3], index=i)
                for i, c in enumerate(all_combos)
            ]

        self.num_strategies = len(self.strategies)

    def get_strategy(self, index: int) -> Strategy:
        """Get strategy by index."""
        return self.strategies[index]

    def get_random_strategy(self) -> Strategy:
        """Get a random strategy."""
        import random
        return random.choice(self.strategies)
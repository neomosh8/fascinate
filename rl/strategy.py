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
        """Convert strategy to simplified GPT system prompt."""
        return (
            f"Adopt a {self.tone} tone when discussing {self.topic}. "
            f"Express {self.emotion} through your word choice and phrasing. "
            f"Begin your response using the '{self.hook}' approach - "
            f"start with a {self.hook} and weave it naturally into your message. "
            f"Use natural speech patterns with appropriate pauses, emphasis through word choice, "
            f"and conversational fillers like 'you know', 'well', 'actually', etc. "
            f"Keep responses under 180 tokens and end with something that invites further conversation."
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
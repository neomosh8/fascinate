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
        """Convert strategy to GPT system prompt with Eleven v3 audio tag guidance."""
        return (
            f"Talk about {self.topic} with a {self.tone} tone. "
            f"Express {self.emotion} in your voice—wrap emotional expressions in square brackets, e.g., [excited], [frustrated sigh], [laughs]. "
            f"Use the hook '{self.hook}' to start your speech and weave it naturally into the story. "
            f"Structure your script with realistic pauses (use ellipses '...'), emphasize important phrases with CAPITALS, and control rhythm with punctuation. "
            f"Example:\n\n"
            f"[{self.emotion}] {self.hook}... I can't believe we're finally talking about {self.topic}. "
            f"It's been on my mind for days! [sighs] The whole thing just keeps evolving. "
            f"You know, some people don't realize how {self.tone.lower()} this gets... but it really does.\n\n"
            f"Try using expressive voice tags like [laughs], [whispers], [sarcastic], or [strong X accent] to convey mood and context. "
            f"Here's a sample:\n\n"
            f"[curious] So, what if I told you that everything you know about {self.topic}... might be wrong?\n"
            f"[whispers] Yeah, it's that serious.\n"
            f"[excited] But don’t worry—we’re diving deep today!\n"
            f"[strong French accent] 'Zat's life, my friend — you can't control everysing.'\n\n"
            f"Wrap each speaker cue or emotion in brackets for Eleven v3 compatibility. Ensure prompt length is at least 250 characters for consistency."
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
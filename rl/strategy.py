"""Communication strategy definitions and management."""

from dataclasses import dataclass
from typing import List, Tuple
import json
import itertools
from config import TONES, TOPICS, EMOTIONS, HOOKS


@dataclass
class Strategy:
    """Represents a communication strategy with memory of good/bad examples."""
    tone: str
    topic: str
    emotion: str
    hook: str
    index: int

    def __post_init__(self):
        # Memory of best/worst responses for this strategy
        self.best_examples: List[Tuple[str, float]] = []
        self.worst_examples: List[Tuple[str, float]] = []
        self.max_examples: int = 5

    def to_prompt(self) -> str:
        """Convert strategy to simplified GPT system prompt."""
        return (
            "You are a mental wellness companion providing supportive, therapy-style conversation. "
            f"Adopt a {self.tone} tone and focus on {self.topic}. "
            f"Express yourself in a {self.emotion} manner. "
            f"Begin with '{self.hook}' or weave it naturally into your response. "
            "Use short, clear sentences with reflective listening and open-ended questions. "
            "and conversational fillers like 'you know', 'well', 'actually', etc. according to tone and emotion you are set to do"
            "Avoid medical advice or diagnoses. Continue gently even if the user is silent."
        )

    def to_key(self) -> str:
        """Return a compact key for dictionary indexing."""
        return f"{self.tone}|{self.topic}|{self.emotion}|{self.hook}"

    @classmethod
    def from_key(cls, key: str) -> 'Strategy':
        """Reconstruct a Strategy from a key string."""
        tone, topic, emotion, hook = key.split('|')
        return cls(tone=tone, topic=topic, emotion=emotion, hook=hook, index=0)

    def add_example(self, response: str, engagement_delta: float):
        """Add a response example with engagement result to memory."""
        example = (response, engagement_delta)
        if engagement_delta > 0.1:
            self.best_examples.append(example)
            self.best_examples.sort(key=lambda x: x[1], reverse=True)
            self.best_examples = self.best_examples[: self.max_examples]
        elif engagement_delta < -0.1:
            self.worst_examples.append(example)
            self.worst_examples.sort(key=lambda x: x[1])
            self.worst_examples = self.worst_examples[: self.max_examples]

    def to_prompt_with_memory(self) -> str:
        """Generate system prompt including memory of past examples."""
        base_prompt = self.to_prompt()
        if not self.best_examples and not self.worst_examples:
            return base_prompt

        memory_prompt = "\n\nBased on past interactions:"

        if self.best_examples:
            memory_prompt += "\n\nExamples that worked VERY WELL (high engagement):"
            for resp, delta in self.best_examples[:5]:
                truncated = resp[:150] + "..." if len(resp) > 150 else resp
                memory_prompt += f"\n- {resp} (engagement +{delta:.2f})"

        if self.worst_examples:
            memory_prompt += "\n\nExamples that did NOT work (low engagement):"
            for resp, delta in self.worst_examples[:3]:
                truncated = resp[:150] + "..." if len(resp) > 150 else resp
                memory_prompt += f"\n- {resp} (engagement {delta:.2f})"

        memory_prompt += (
            "\n\nUse the successful examples as inspiration for style and content or even exploiting or going further and deepen it  "
            "Avoid patterns from unsuccessful examples."
        )

        return base_prompt + memory_prompt


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

    def save_memory(self, filepath):
        """Persist best and worst examples for all strategies."""
        memory_data = {}
        for strategy in self.strategies:
            if strategy.best_examples or strategy.worst_examples:
                memory_data[strategy.index] = {
                    "best": strategy.best_examples,
                    "worst": strategy.worst_examples,
                }

        with open(filepath, "w") as f:
            json.dump(memory_data, f, indent=2)

    def load_memory(self, filepath):
        """Load best and worst examples from disk if available."""
        from pathlib import Path

        path = Path(filepath)
        if not path.exists():
            return

        with open(path, "r") as f:
            memory_data = json.load(f)

        for idx_str, examples in memory_data.items():
            idx = int(idx_str)
            if 0 <= idx < len(self.strategies):
                self.strategies[idx].best_examples = examples.get("best", [])
                self.strategies[idx].worst_examples = examples.get("worst", [])


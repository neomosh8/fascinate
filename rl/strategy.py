"""Communication strategy definitions and management."""

from dataclasses import dataclass
from typing import List, Tuple, Dict
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
            f"Adopt a {self.tone} tone "
            f"talk {self.topic} language  "
            f"Express {self.emotion} through your word choice and phrasing. "
            f"start with a {self.hook} or use it and weave it naturally into your message. "
            f"Use natural speech patterns with appropriate pauses, emphasis through word choice, "
            f"and conversational fillers like 'you know', 'well', 'actually', etc. according to tone and emotion you are set to do"
            f"for the topic of conversation maintain what user wants through the session. don't break topic or subject. continue the topic as it goes to real quality conversation"
            f"don't mention if the user is silent, go on"
        )

    def to_key(self) -> str:
        """Return a compact key for dictionary indexing."""
        return f"{self.tone}|{self.topic}|{self.emotion}|{self.hook}"

    # Add this to rl/strategy.py in the Strategy class
    def get_emotion_adapted_tts_params(self, user_emotion: float, user_engagement: float) -> Dict[str, float]:
        """Get TTS parameters adapted for user's emotional state."""

        # Base parameters
        base_speed = 1.0
        base_pitch = 0.0
        base_energy = 1.0
        base_warmth = 1.0

        # Adjust based on strategy tone
        if self.tone in ["calm", "empathetic", "gentle"]:
            base_speed *= 0.9
            base_warmth *= 1.2
        elif self.tone in ["confident", "professional"]:
            base_speed *= 1.1
            base_energy *= 1.1
        elif self.tone in ["playful", "friendly"]:
            base_speed *= 1.05
            base_energy *= 1.1
            base_warmth *= 1.1

        # Adjust based on strategy emotion
        if self.emotion in ["whisper", "sad", "thoughtful"]:
            base_speed *= 0.85
            base_energy *= 0.8
        elif self.emotion in ["happy", "excited"]:
            base_speed *= 1.15
            base_energy *= 1.2
        elif self.emotion in ["angry", "serious"]:
            base_energy *= 1.1
            base_pitch -= 0.1

        # Adapt to user state
        if user_emotion < 0.4:  # User seems down
            base_speed *= 0.9
            base_warmth *= 1.3
        elif user_emotion > 0.6:  # User seems positive
            base_energy *= 1.1

        if user_engagement < 0.4:  # User disengaged
            base_energy *= 1.2
            base_speed *= 1.1

        return {
            "speed": max(0.7, min(1.3, base_speed)),
            "pitch": max(-0.3, min(0.3, base_pitch)),
            "energy": max(0.7, min(1.3, base_energy)),
            "warmth": max(0.7, min(1.3, base_warmth)),
        }
    @classmethod
    def from_key(cls, key: str) -> 'Strategy':
        """Reconstruct a Strategy from a key string."""
        tone, topic, emotion, hook = key.split('|')
        return cls(tone=tone, topic=topic, emotion=emotion, hook=hook, index=0)

    def add_example(self, response: str, engagement_delta: float):
        """Add a response example with engagement result to memory."""
        example = (response, engagement_delta)
        if engagement_delta > 0.2:
            self.best_examples.append(example)
            self.best_examples.sort(key=lambda x: x[1], reverse=True)
            self.best_examples = self.best_examples[: self.max_examples]
        elif engagement_delta < -0.2:
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
            memory_prompt += "\n\nExamples that worked VERY WELL, continue on this line (high engagement):"
            for resp, delta in self.best_examples[:5]:
                truncated = resp[:150] + "..." if len(resp) > 150 else resp
                memory_prompt += f"\n- {resp} (engagement +{delta:.2f})"

        if self.worst_examples:
            memory_prompt += "\n\nExamples that did NOT work, do not continue on this line, acknowledge it was not right(low engagement):"
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

    def get_mutated_strategy(self, base_strategy: Strategy) -> Strategy:
        """Return a slight variation of the given strategy."""
        import random
        tone = base_strategy.tone
        topic = base_strategy.topic
        emotion = base_strategy.emotion
        hook = base_strategy.hook

        component = random.choice(["tone", "topic", "emotion", "hook"])
        if component == "tone":
            tone = random.choice(TONES)
        elif component == "topic":
            topic = random.choice(TOPICS)
        elif component == "emotion":
            emotion = random.choice(EMOTIONS)
        else:
            hook = random.choice(HOOKS)

        return Strategy(tone=tone, topic=topic, emotion=emotion, hook=hook, index=-1)

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


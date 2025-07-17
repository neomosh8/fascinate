# strategy_system.py
import random
from dataclasses import dataclass
from typing import List, Dict
import itertools


@dataclass
class CommunicationStrategy:
    tone: str
    topic: str
    emotion: str
    hook: str

    def to_prompt(self, user_name: str = "friend") -> str:
        """Convert strategy to AI instruction prompt"""
        hook_text = self.hook.replace("[name]", user_name)

        prompt = f"""
Respond using this communication strategy:
- Talk about {self.topic} with a {self.tone} tone
- Express {self.emotion} in your voice
- Start with the hook: "{hook_text}"
- Keep response conversational and engaging (30-60 seconds when spoken)
- Match the emotional energy level requested
"""
        return prompt


class StrategyGenerator:
    """Generates and manages communication strategies"""

    def __init__(self):
        self.tones = ["playful", "naughty", "informational", "bossy", "aggressive", "sarcastic"]
        self.topics = ["politics", "facts", "story", "controversial", "dad joke", "flirting"]
        self.emotions = ["happy", "sad", "serious", "scared", "whisper", "shout out", "laughter"]
        self.hooks = ["hey [name]", "you know what?", "are you with me?", "listen", "look"]

        # All possible strategies
        self.all_strategies = list(itertools.product(self.tones, self.topics, self.emotions, self.hooks))
        self.strategy_count = len(self.all_strategies)

    def get_strategy_by_index(self, index: int) -> CommunicationStrategy:
        """Get strategy by index (for RL agent)"""
        strategy_tuple = self.all_strategies[index % self.strategy_count]
        return CommunicationStrategy(*strategy_tuple)

    def get_random_strategy(self) -> CommunicationStrategy:
        """Get random strategy"""
        return CommunicationStrategy(
            tone=random.choice(self.tones),
            topic=random.choice(self.topics),
            emotion=random.choice(self.emotions),
            hook=random.choice(self.hooks)
        )

    def get_strategy_count(self) -> int:
        """Total number of possible strategies"""
        return self.strategy_count
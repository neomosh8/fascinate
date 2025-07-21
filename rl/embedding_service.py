import numpy as np
from typing import List, Optional
from openai import OpenAI
from rl.strategy import Strategy

class EmbeddingService:
    """Service for generating embeddings with caching."""

    def __init__(self):
        self.client = OpenAI()
        self.cache = {}

    def embed_text(self, text: str) -> np.ndarray:
        """Embed any text using OpenAI API with simple caching."""
        if not text.strip():
            return np.zeros(1536)
        if text in self.cache:
            return self.cache[text]
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = np.array(response.data[0].embedding)
        self.cache[text] = embedding
        return embedding


    def embed_strategy(self, strategy: Strategy) -> np.ndarray:
        """Embed strategy represented as text."""
        strategy_text = (
            f"tone:{strategy.tone} topic:{strategy.topic} "
            f"emotion:{strategy.emotion} hook:{strategy.hook}"
        )
        return self.embed_text(strategy_text)

    def create_engagement_features(self, scores: List[float]) -> np.ndarray:
        """Create hand-crafted statistical engagement features."""
        if not scores:
            return np.zeros(6)
        scores_array = np.array(scores)
        features = [
            np.mean(scores_array),
            np.std(scores_array),
            scores_array[-1] if len(scores_array) > 0 else 0.5,
            np.max(scores_array),
            np.min(scores_array),
            scores_array[-1] - scores_array[0] if len(scores_array) > 1 else 0,
        ]
        return np.array(features)


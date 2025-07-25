"""Track concept activation patterns."""

import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict


class ConceptTracker:
    """Track which concepts generate high neural activation."""

    def __init__(self):
        self.concept_activations = defaultdict(list)
        self.concept_mentions = defaultdict(int)
        self.recent_concepts = deque(maxlen=20)

    def record_concept_activation(self, concepts: List[str], engagement_score: float):
        """Record engagement score for mentioned concepts."""
        for concept in concepts:
            self.concept_activations[concept].append(engagement_score)
            self.concept_mentions[concept] += 1
            self.recent_concepts.append((concept, engagement_score))

    def get_hot_concepts(self, min_mentions: int = 2, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get concepts with highest average engagement."""
        concept_scores = []
        for concept, scores in self.concept_activations.items():
            if len(scores) >= min_mentions:
                avg_score = np.mean(scores)
                concept_scores.append((concept, avg_score))
        concept_scores.sort(key=lambda x: x[1], reverse=True)
        return concept_scores[:top_k]

    def should_exploit_concept(self, concept: str, threshold: float = 0.7) -> bool:
        """Check if concept should be exploited (deeply explored)."""
        if concept not in self.concept_activations:
            return False
        scores = self.concept_activations[concept]
        return len(scores) >= 2 and np.mean(scores) >= threshold

    def get_exploration_status(self) -> Dict:
        """Get current exploration status."""
        return {
            'total_concepts': len(self.concept_activations),
            'hot_concepts': self.get_hot_concepts(),
            'recent_activity': list(self.recent_concepts)[-5:],
        }

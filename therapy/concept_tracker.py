"""Track concept activation patterns."""

import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict


class ConceptTracker:
    def __init__(self):
        self.concept_activations = defaultdict(list)
        self.concept_emotions = defaultdict(list)  # NEW: Track emotions per concept
        self.concept_mentions = defaultdict(int)
        self.recent_concepts = deque(maxlen=20)

    def record_concept_activation(self, concepts: List[str], engagement_score: float, emotion_score: float):
        """Record both engagement AND emotion for mentioned concepts."""
        for concept in concepts:
            self.concept_activations[concept].append(engagement_score)
            self.concept_emotions[concept].append(emotion_score)  # NEW
            self.concept_mentions[concept] += 1
            self.recent_concepts.append((concept, engagement_score, emotion_score))  # Updated

    def get_hot_concepts(self, min_mentions: int = 2, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get concepts with highest combined engagement + emotional significance."""
        concept_scores = []
        for concept, eng_scores in self.concept_activations.items():
            if len(eng_scores) >= min_mentions:
                avg_engagement = np.mean(eng_scores)

                # NEW: Calculate emotional intensity (absolute deviation from neutral 0.5)
                emotion_scores = self.concept_emotions[concept]
                emotional_intensity = np.mean([abs(e - 0.5) * 2 for e in emotion_scores])  # 0-1 scale

                # Combined score: engagement weighted by emotional significance
                combined_score = avg_engagement * (1 + emotional_intensity)  # Boost by emotion
                concept_scores.append((concept, combined_score))

        concept_scores.sort(key=lambda x: x[1], reverse=True)
        return concept_scores[:top_k]

    def get_emotional_profile(self, concept: str) -> Dict:
        """Get emotional profile for a concept."""
        if concept not in self.concept_emotions:
            return {}

        emotions = self.concept_emotions[concept]
        return {
            'avg_emotion': np.mean(emotions),
            'emotional_intensity': np.mean([abs(e - 0.5) * 2 for e in emotions]),
            'emotional_consistency': 1 - np.std(emotions),  # Low std = consistent emotion
            'valence': 'positive' if np.mean(emotions) > 0.5 else 'negative'
        }
"""Embedding-based repetition detector for analysis turns."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class RepetitionDetector:
    """Detects when agents start repeating themselves using cosine similarity."""

    def __init__(self, threshold: float = 0.85, max_consecutive: int = 3, model_name: str = "all-MiniLM-L6-v2"):
        self.threshold = threshold
        self.max_consecutive = max_consecutive
        self.model = SentenceTransformer(model_name, device="cpu")
        self._embeddings: dict[str, list] = {}  # agent -> list of embeddings
        self._consecutive_flags: dict[str, int] = {}

    def check(self, agent: str, text: str) -> tuple[bool, float]:
        """Check if a turn is semantically repetitive.

        Returns (is_repetitive, max_similarity_score).
        """
        embedding = self.model.encode([text])[0]

        if agent not in self._embeddings:
            self._embeddings[agent] = []
            self._consecutive_flags[agent] = 0

        if not self._embeddings[agent]:
            self._embeddings[agent].append(embedding)
            return False, 0.0

        # Compute cosine similarity against all prior turns from this agent
        prior = np.array(self._embeddings[agent])
        similarities = np.dot(prior, embedding) / (
            np.linalg.norm(prior, axis=1) * np.linalg.norm(embedding)
        )
        max_sim = float(np.max(similarities))

        is_repetitive = max_sim > self.threshold
        if is_repetitive:
            self._consecutive_flags[agent] += 1
            logger.warning(f"{agent} repetition detected (sim={max_sim:.3f}, consecutive={self._consecutive_flags[agent]})")
        else:
            self._consecutive_flags[agent] = 0

        self._embeddings[agent].append(embedding)
        return is_repetitive, max_sim

    def should_terminate(self) -> bool:
        """Check if any agent has exceeded the max consecutive repetition threshold."""
        return any(count >= self.max_consecutive for count in self._consecutive_flags.values())

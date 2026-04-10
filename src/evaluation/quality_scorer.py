"""Scientific analysis quality scorer — LLM-as-judge for round quality."""

import json
import logging

from src.prompts.templates import QUALITY_SCORING

logger = logging.getLogger(__name__)


class QualityScorer:
    """Scores analysis rounds on novelty, rigor, engagement, and depth."""

    def __init__(self, stagnation_threshold: float = 2.5, max_stagnant_rounds: int = 3):
        self.stagnation_threshold = stagnation_threshold
        self.max_stagnant_rounds = max_stagnant_rounds
        self.round_scores = []
        self._consecutive_low = 0

    def score_round(self, round_text: str, tabby_client) -> dict:
        """Score a round using the currently loaded model as judge.

        Returns dict with novelty, rigor, engagement, depth scores (1-5 each).
        """
        prompt = QUALITY_SCORING.format(round_text=round_text)
        messages = [
            {"role": "system", "content": "You are a scientific quality evaluator. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = tabby_client.chat_completion(messages, temperature=0.2, max_tokens=256)
            scores = json.loads(raw.strip())
            self.round_scores.append(scores)

            avg = sum(scores.values()) / len(scores)
            if avg < self.stagnation_threshold:
                self._consecutive_low += 1
            else:
                self._consecutive_low = 0

            logger.info(f"Round quality: {scores} (avg={avg:.1f}, consecutive_low={self._consecutive_low})")
            return scores
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Quality scoring failed: {e}")
            return {"novelty": 3, "rigor": 3, "engagement": 3, "depth": 3}

    def should_terminate(self) -> bool:
        """Check if analysis has stagnated."""
        return self._consecutive_low >= self.max_stagnant_rounds

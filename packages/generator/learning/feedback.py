"""Feedback collection for card generation quality improvement."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final

_DEFAULT_MAX_ENTRIES: Final = 1000


@dataclass(frozen=True, slots=True)
class QualityFeedback:
    """Feedback on a generated card's quality."""

    topic: str
    card_id: str
    quality_score: float
    issues: tuple[str, ...] = ()
    strengths: tuple[str, ...] = ()
    timestamp: float = 0.0


class FeedbackCollector:
    """Records what worked and what didn't for future improvement.

    Simple in-memory store with FIFO eviction.
    """

    def __init__(self, *, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._feedback: list[QualityFeedback] = []
        self._max_entries = max_entries

    def record(self, feedback: QualityFeedback) -> None:
        """Record quality feedback, evicting oldest if over limit."""
        self._feedback.append(feedback)
        if len(self._feedback) > self._max_entries:
            self._feedback = self._feedback[-self._max_entries :]

    def summary(self, topic: str | None = None) -> dict[str, object]:
        """Get feedback summary: avg quality, common issues, common strengths."""
        return {
            "average_quality": self.average_quality(topic),
            "common_issues": self.common_issues(topic),
            "total_feedback": len(self._filter(topic)),
        }

    def common_issues(self, topic: str | None = None, *, limit: int = 5) -> tuple[str, ...]:
        """Get most frequent issues."""
        entries = self._filter(topic)
        counter: Counter[str] = Counter()
        for fb in entries:
            counter.update(fb.issues)
        return tuple(issue for issue, _ in counter.most_common(limit))

    def average_quality(self, topic: str | None = None) -> float:
        """Get average quality score."""
        entries = self._filter(topic)
        if not entries:
            return 0.0
        return sum(fb.quality_score for fb in entries) / len(entries)

    def clear(self) -> None:
        """Clear all stored feedback."""
        self._feedback.clear()

    def _filter(self, topic: str | None) -> list[QualityFeedback]:
        if topic is None:
            return self._feedback
        return [fb for fb in self._feedback if fb.topic == topic]

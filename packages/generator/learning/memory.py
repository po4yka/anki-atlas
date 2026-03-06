"""Generation outcome memory for learning from past results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

_DEFAULT_MAX_ENTRIES: Final = 1000


@dataclass(frozen=True, slots=True)
class GenerationOutcome:
    """Record of a single generation attempt."""

    topic: str
    model_used: str
    card_count: int
    quality_score: float
    success: bool
    timestamp: float
    warnings: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


class GenerationMemory:
    """Stores past generation outcomes for learning.

    Simple in-memory store with FIFO eviction. No external dependencies.
    """

    def __init__(self, *, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._outcomes: list[GenerationOutcome] = []
        self._max_entries = max_entries

    def record(self, outcome: GenerationOutcome) -> None:
        """Record a generation outcome, evicting oldest if over limit."""
        self._outcomes.append(outcome)
        if len(self._outcomes) > self._max_entries:
            self._outcomes = self._outcomes[-self._max_entries :]

    def outcomes_for_topic(
        self, topic: str, *, limit: int = 10
    ) -> tuple[GenerationOutcome, ...]:
        """Get recent outcomes for a topic, newest first."""
        matches = [o for o in reversed(self._outcomes) if o.topic == topic]
        return tuple(matches[:limit])

    def average_quality(self, topic: str | None = None) -> float:
        """Get average quality score, optionally filtered by topic."""
        outcomes = self._filter(topic)
        if not outcomes:
            return 0.0
        return sum(o.quality_score for o in outcomes) / len(outcomes)

    def success_rate(self, topic: str | None = None) -> float:
        """Get success rate, optionally filtered by topic."""
        outcomes = self._filter(topic)
        if not outcomes:
            return 0.0
        return sum(1 for o in outcomes if o.success) / len(outcomes)

    def stats(self) -> dict[str, object]:
        """Get summary statistics."""
        topics = {o.topic for o in self._outcomes}
        return {
            "total_outcomes": len(self._outcomes),
            "success_rate": self.success_rate(),
            "average_quality": self.average_quality(),
            "topics_seen": len(topics),
        }

    def clear(self) -> None:
        """Clear all stored outcomes."""
        self._outcomes.clear()

    def _filter(self, topic: str | None) -> list[GenerationOutcome]:
        if topic is None:
            return self._outcomes
        return [o for o in self._outcomes if o.topic == topic]

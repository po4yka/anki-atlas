"""Few-shot example management for card generation prompting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Final

_DEFAULT_MAX_PER_TOPIC: Final = 20


@dataclass(frozen=True, slots=True)
class FewShotExample:
    """A curated example card for few-shot prompting."""

    topic: str
    question: str
    answer: str
    quality_score: float
    tags: tuple[str, ...] = ()
    source: str = ""


class FewShotStore:
    """Manages curated few-shot examples for prompting.

    In-memory store keyed by topic. Keeps top examples by quality.
    """

    def __init__(self, *, max_examples_per_topic: int = _DEFAULT_MAX_PER_TOPIC) -> None:
        self._examples: dict[str, list[FewShotExample]] = defaultdict(list)
        self._max_per_topic = max_examples_per_topic

    def add(self, example: FewShotExample) -> None:
        """Add a few-shot example. Evicts lowest quality if over limit."""
        bucket = self._examples[example.topic]
        bucket.append(example)
        if len(bucket) > self._max_per_topic:
            bucket.sort(key=lambda e: e.quality_score)
            bucket.pop(0)

    def get(
        self, topic: str, *, limit: int = 3, min_quality: float = 0.0
    ) -> tuple[FewShotExample, ...]:
        """Get best examples for a topic, sorted by quality descending."""
        bucket = self._examples.get(topic, [])
        filtered = [e for e in bucket if e.quality_score >= min_quality]
        filtered.sort(key=lambda e: e.quality_score, reverse=True)
        return tuple(filtered[:limit])

    def topics(self) -> tuple[str, ...]:
        """List all topics with examples."""
        return tuple(sorted(self._examples.keys()))

    def remove(self, topic: str, question: str) -> bool:
        """Remove a specific example by topic and question. Returns True if found."""
        bucket = self._examples.get(topic, [])
        for i, ex in enumerate(bucket):
            if ex.question == question:
                bucket.pop(i)
                if not bucket:
                    del self._examples[topic]
                return True
        return False

    def count(self, topic: str | None = None) -> int:
        """Count examples, optionally filtered by topic."""
        if topic is not None:
            return len(self._examples.get(topic, []))
        return sum(len(v) for v in self._examples.values())

    def clear(self) -> None:
        """Clear all stored examples."""
        self._examples.clear()

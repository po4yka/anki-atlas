"""Tests for packages.generator.learning modules."""

from __future__ import annotations

import time

from packages.generator.learning.examples import FewShotExample, FewShotStore
from packages.generator.learning.feedback import FeedbackCollector, QualityFeedback
from packages.generator.learning.memory import GenerationMemory, GenerationOutcome

# --- GenerationOutcome ---


class TestGenerationOutcome:
    def test_creation(self) -> None:
        outcome = GenerationOutcome(
            topic="python",
            model_used="gpt-4",
            card_count=5,
            quality_score=0.85,
            success=True,
            timestamp=1000.0,
        )
        assert outcome.topic == "python"
        assert outcome.model_used == "gpt-4"
        assert outcome.card_count == 5
        assert outcome.quality_score == 0.85
        assert outcome.success is True

    def test_frozen(self) -> None:
        outcome = GenerationOutcome(
            topic="python",
            model_used="gpt-4",
            card_count=5,
            quality_score=0.85,
            success=True,
            timestamp=1000.0,
        )
        try:
            outcome.topic = "java"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

    def test_defaults(self) -> None:
        outcome = GenerationOutcome(
            topic="math",
            model_used="gpt-4",
            card_count=1,
            quality_score=0.5,
            success=True,
            timestamp=0.0,
        )
        assert outcome.warnings == ()
        assert outcome.metadata == {}


# --- GenerationMemory ---


class TestGenerationMemory:
    def _make_outcome(
        self,
        topic: str = "python",
        quality: float = 0.8,
        success: bool = True,
        ts: float = 0.0,
    ) -> GenerationOutcome:
        return GenerationOutcome(
            topic=topic,
            model_used="gpt-4",
            card_count=3,
            quality_score=quality,
            success=success,
            timestamp=ts,
        )

    def test_record_and_retrieve(self) -> None:
        mem = GenerationMemory()
        o = self._make_outcome()
        mem.record(o)
        results = mem.outcomes_for_topic("python")
        assert len(results) == 1
        assert results[0] is o

    def test_outcomes_for_topic_newest_first(self) -> None:
        mem = GenerationMemory()
        o1 = self._make_outcome(ts=1.0)
        o2 = self._make_outcome(ts=2.0)
        mem.record(o1)
        mem.record(o2)
        results = mem.outcomes_for_topic("python")
        assert results[0].timestamp == 2.0
        assert results[1].timestamp == 1.0

    def test_outcomes_for_topic_limit(self) -> None:
        mem = GenerationMemory()
        for i in range(10):
            mem.record(self._make_outcome(ts=float(i)))
        results = mem.outcomes_for_topic("python", limit=3)
        assert len(results) == 3

    def test_average_quality(self) -> None:
        mem = GenerationMemory()
        mem.record(self._make_outcome(quality=0.6))
        mem.record(self._make_outcome(quality=0.8))
        assert mem.average_quality() == 0.7

    def test_average_quality_by_topic(self) -> None:
        mem = GenerationMemory()
        mem.record(self._make_outcome(topic="python", quality=0.8))
        mem.record(self._make_outcome(topic="math", quality=0.4))
        assert mem.average_quality("python") == 0.8

    def test_average_quality_empty(self) -> None:
        mem = GenerationMemory()
        assert mem.average_quality() == 0.0

    def test_success_rate(self) -> None:
        mem = GenerationMemory()
        mem.record(self._make_outcome(success=True))
        mem.record(self._make_outcome(success=False))
        assert mem.success_rate() == 0.5

    def test_success_rate_empty(self) -> None:
        mem = GenerationMemory()
        assert mem.success_rate() == 0.0

    def test_stats(self) -> None:
        mem = GenerationMemory()
        mem.record(self._make_outcome(topic="python"))
        mem.record(self._make_outcome(topic="math"))
        stats = mem.stats()
        assert stats["total_outcomes"] == 2
        assert stats["topics_seen"] == 2

    def test_max_entries_eviction(self) -> None:
        mem = GenerationMemory(max_entries=3)
        for i in range(5):
            mem.record(self._make_outcome(ts=float(i)))
        results = mem.outcomes_for_topic("python")
        assert len(results) == 3
        assert results[0].timestamp == 4.0  # newest
        assert results[2].timestamp == 2.0  # oldest remaining

    def test_clear(self) -> None:
        mem = GenerationMemory()
        mem.record(self._make_outcome())
        mem.clear()
        assert mem.outcomes_for_topic("python") == ()


# --- FewShotExample ---


class TestFewShotExample:
    def test_creation(self) -> None:
        ex = FewShotExample(
            topic="python",
            question="What is a list?",
            answer="An ordered mutable collection.",
            quality_score=0.9,
        )
        assert ex.topic == "python"
        assert ex.quality_score == 0.9

    def test_frozen(self) -> None:
        ex = FewShotExample(
            topic="python",
            question="Q",
            answer="A",
            quality_score=0.5,
        )
        try:
            ex.topic = "java"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

    def test_defaults(self) -> None:
        ex = FewShotExample(
            topic="math",
            question="Q",
            answer="A",
            quality_score=0.5,
        )
        assert ex.tags == ()
        assert ex.source == ""


# --- FewShotStore ---


class TestFewShotStore:
    def _make_example(
        self, topic: str = "python", question: str = "Q", quality: float = 0.8
    ) -> FewShotExample:
        return FewShotExample(
            topic=topic,
            question=question,
            answer="A",
            quality_score=quality,
        )

    def test_add_and_get(self) -> None:
        store = FewShotStore()
        ex = self._make_example()
        store.add(ex)
        results = store.get("python")
        assert len(results) == 1
        assert results[0] is ex

    def test_get_sorted_by_quality(self) -> None:
        store = FewShotStore()
        store.add(self._make_example(question="Q1", quality=0.5))
        store.add(self._make_example(question="Q2", quality=0.9))
        store.add(self._make_example(question="Q3", quality=0.7))
        results = store.get("python", limit=3)
        assert results[0].quality_score == 0.9
        assert results[1].quality_score == 0.7
        assert results[2].quality_score == 0.5

    def test_get_min_quality(self) -> None:
        store = FewShotStore()
        store.add(self._make_example(question="Q1", quality=0.3))
        store.add(self._make_example(question="Q2", quality=0.8))
        results = store.get("python", min_quality=0.5)
        assert len(results) == 1
        assert results[0].quality_score == 0.8

    def test_topics(self) -> None:
        store = FewShotStore()
        store.add(self._make_example(topic="python"))
        store.add(self._make_example(topic="math"))
        assert store.topics() == ("math", "python")

    def test_remove(self) -> None:
        store = FewShotStore()
        store.add(self._make_example(question="Q1"))
        assert store.remove("python", "Q1") is True
        assert store.count("python") == 0

    def test_remove_nonexistent(self) -> None:
        store = FewShotStore()
        assert store.remove("python", "Q1") is False

    def test_count(self) -> None:
        store = FewShotStore()
        store.add(self._make_example(topic="python", question="Q1"))
        store.add(self._make_example(topic="python", question="Q2"))
        store.add(self._make_example(topic="math", question="Q1"))
        assert store.count("python") == 2
        assert store.count() == 3

    def test_max_examples_eviction(self) -> None:
        store = FewShotStore(max_examples_per_topic=2)
        store.add(self._make_example(question="Q1", quality=0.5))
        store.add(self._make_example(question="Q2", quality=0.9))
        store.add(self._make_example(question="Q3", quality=0.7))
        assert store.count("python") == 2
        results = store.get("python", limit=10)
        qualities = {r.quality_score for r in results}
        assert 0.5 not in qualities  # lowest evicted

    def test_clear(self) -> None:
        store = FewShotStore()
        store.add(self._make_example())
        store.clear()
        assert store.count() == 0

    def test_remove_cleans_empty_topic(self) -> None:
        store = FewShotStore()
        store.add(self._make_example(topic="python", question="Q"))
        store.remove("python", "Q")
        assert "python" not in store.topics()


# --- QualityFeedback ---


class TestQualityFeedback:
    def test_creation(self) -> None:
        fb = QualityFeedback(
            topic="python",
            card_id="card-1",
            quality_score=0.8,
            issues=("too long",),
            strengths=("clear",),
        )
        assert fb.topic == "python"
        assert fb.card_id == "card-1"
        assert fb.issues == ("too long",)

    def test_frozen(self) -> None:
        fb = QualityFeedback(
            topic="python",
            card_id="card-1",
            quality_score=0.5,
        )
        try:
            fb.topic = "java"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

    def test_defaults(self) -> None:
        fb = QualityFeedback(
            topic="math",
            card_id="card-1",
            quality_score=0.5,
        )
        assert fb.issues == ()
        assert fb.strengths == ()
        assert fb.timestamp == 0.0


# --- FeedbackCollector ---


class TestFeedbackCollector:
    def _make_feedback(
        self,
        topic: str = "python",
        quality: float = 0.8,
        issues: tuple[str, ...] = (),
    ) -> QualityFeedback:
        return QualityFeedback(
            topic=topic,
            card_id="card-1",
            quality_score=quality,
            issues=issues,
            timestamp=time.time(),
        )

    def test_record_and_summary(self) -> None:
        collector = FeedbackCollector()
        collector.record(self._make_feedback(quality=0.8))
        collector.record(self._make_feedback(quality=0.6))
        summary = collector.summary()
        assert summary["total_feedback"] == 2
        assert summary["average_quality"] == 0.7

    def test_common_issues(self) -> None:
        collector = FeedbackCollector()
        collector.record(self._make_feedback(issues=("too long", "unclear")))
        collector.record(self._make_feedback(issues=("too long",)))
        collector.record(self._make_feedback(issues=("unclear",)))
        issues = collector.common_issues()
        assert issues[0] == "too long"  # most frequent
        assert "unclear" in issues

    def test_common_issues_limit(self) -> None:
        collector = FeedbackCollector()
        for i in range(10):
            collector.record(self._make_feedback(issues=(f"issue-{i}",)))
        issues = collector.common_issues(limit=3)
        assert len(issues) == 3

    def test_average_quality_by_topic(self) -> None:
        collector = FeedbackCollector()
        collector.record(self._make_feedback(topic="python", quality=0.8))
        collector.record(self._make_feedback(topic="math", quality=0.4))
        assert collector.average_quality("python") == 0.8

    def test_average_quality_empty(self) -> None:
        collector = FeedbackCollector()
        assert collector.average_quality() == 0.0

    def test_max_entries_eviction(self) -> None:
        collector = FeedbackCollector(max_entries=3)
        for i in range(5):
            collector.record(self._make_feedback(quality=float(i) / 10))
        summary = collector.summary()
        assert summary["total_feedback"] == 3

    def test_clear(self) -> None:
        collector = FeedbackCollector()
        collector.record(self._make_feedback())
        collector.clear()
        assert collector.average_quality() == 0.0

    def test_summary_by_topic(self) -> None:
        collector = FeedbackCollector()
        collector.record(self._make_feedback(topic="python"))
        collector.record(self._make_feedback(topic="math"))
        summary = collector.summary("python")
        assert summary["total_feedback"] == 1

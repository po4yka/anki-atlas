"""Tests for MCP formatters.

Tool tests are kept minimal since they rely on lazy imports and
complex service mocking. The formatters contain the core output logic
and are tested thoroughly.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest


# Mock dataclasses matching the real service types
@dataclass
class MockSearchResult:
    note_id: int
    rrf_score: float
    semantic_score: float | None = None
    semantic_rank: int | None = None
    fts_score: float | None = None
    fts_rank: int | None = None
    headline: str | None = None

    @property
    def sources(self) -> list[str]:
        sources = []
        if self.semantic_rank is not None:
            sources.append("semantic")
        if self.fts_rank is not None:
            sources.append("fts")
        return sources


@dataclass
class MockFusionStats:
    semantic_only: int = 0
    fts_only: int = 0
    both: int = 0
    total: int = 0


@dataclass
class MockHybridSearchResult:
    results: list[MockSearchResult]
    stats: MockFusionStats
    query: str
    filters_applied: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockNoteDetail:
    note_id: int
    model_id: int
    normalized_text: str
    tags: list[str]
    deck_names: list[str]
    mature: bool
    lapses: int
    reps: int


@dataclass
class MockTopicCoverage:
    topic_id: int
    path: str
    label: str
    note_count: int = 0
    subtree_count: int = 0
    child_count: int = 0
    covered_children: int = 0
    mature_count: int = 0
    avg_confidence: float = 0.0
    weak_notes: int = 0
    avg_lapses: float = 0.0


@dataclass
class MockTopicGap:
    topic_id: int
    path: str
    label: str
    description: str | None
    gap_type: str
    note_count: int
    threshold: int
    nearest_notes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MockDuplicateCluster:
    representative_id: int
    representative_text: str
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    deck_names: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return 1 + len(self.duplicates)


@dataclass
class MockDuplicateStats:
    notes_scanned: int = 0
    clusters_found: int = 0
    total_duplicates: int = 0
    avg_cluster_size: float = 0.0


@dataclass
class MockSyncStats:
    decks_upserted: int = 0
    models_upserted: int = 0
    notes_upserted: int = 0
    notes_deleted: int = 0
    cards_upserted: int = 0
    card_stats_upserted: int = 0
    duration_ms: int = 0


class TestFormatSearchResult:
    """Tests for format_search_result."""

    def test_empty_results(self) -> None:
        from apps.mcp.formatters import format_search_result

        result = MockHybridSearchResult(
            results=[],
            stats=MockFusionStats(total=0),
            query="test query",
        )
        output = format_search_result(result, {})

        assert "Search Results for: test query" in output
        assert "No results found" in output

    def test_with_results(self) -> None:
        from apps.mcp.formatters import format_search_result

        results = [
            MockSearchResult(
                note_id=1,
                rrf_score=0.95,
                semantic_rank=1,
                fts_rank=2,
            ),
            MockSearchResult(
                note_id=2,
                rrf_score=0.85,
                semantic_rank=3,
            ),
        ]
        stats = MockFusionStats(semantic_only=1, fts_only=0, both=1, total=2)
        result = MockHybridSearchResult(
            results=results,
            stats=stats,
            query="calculus",
        )
        note_details = {
            1: MockNoteDetail(
                note_id=1,
                model_id=100,
                normalized_text="What is the derivative of x^2?",
                tags=["math", "calculus"],
                deck_names=["Math::Calculus"],
                mature=True,
                lapses=2,
                reps=10,
            ),
            2: MockNoteDetail(
                note_id=2,
                model_id=100,
                normalized_text="Explain the chain rule",
                tags=["math"],
                deck_names=["Math"],
                mature=False,
                lapses=5,
                reps=8,
            ),
        }

        output = format_search_result(result, note_details)

        assert "Search Results for: calculus" in output
        assert "Found **2** results" in output
        assert "0.950" in output
        assert "derivative" in output
        assert "math, calculus" in output

    def test_with_filters_and_results(self) -> None:
        from apps.mcp.formatters import format_search_result

        result = MockHybridSearchResult(
            results=[MockSearchResult(note_id=1, rrf_score=0.9, semantic_rank=1)],
            stats=MockFusionStats(total=1, semantic_only=1),
            query="test",
            filters_applied={"deck_names": ["Math"], "tags": ["important"]},
        )
        note_details = {
            1: MockNoteDetail(
                note_id=1,
                model_id=1,
                normalized_text="Test",
                tags=[],
                deck_names=["Math"],
                mature=False,
                lapses=0,
                reps=0,
            )
        }
        output = format_search_result(result, note_details)

        assert "Filters:" in output
        assert "deck_names" in output

    def test_truncates_long_results(self) -> None:
        from apps.mcp.formatters import format_search_result

        results = [
            MockSearchResult(note_id=i, rrf_score=1.0 - i * 0.01, semantic_rank=i)
            for i in range(1, 30)
        ]
        stats = MockFusionStats(total=29, semantic_only=29)
        result = MockHybridSearchResult(results=results, stats=stats, query="test")

        output = format_search_result(result, {})

        assert "...and 9 more results" in output


class TestFormatCoverageResult:
    """Tests for format_coverage_result."""

    def test_basic_coverage(self) -> None:
        from apps.mcp.formatters import format_coverage_result

        coverage = MockTopicCoverage(
            topic_id=1,
            path="math/calculus",
            label="Calculus",
            note_count=50,
            subtree_count=150,
            child_count=5,
            covered_children=3,
            mature_count=30,
            avg_confidence=0.85,
            weak_notes=5,
            avg_lapses=2.3,
        )

        output = format_coverage_result(coverage)

        assert "Coverage: Calculus" in output
        assert "math/calculus" in output
        assert "**Direct notes**: 50" in output
        assert "**Mature notes**: 30" in output
        assert "3/5" in output
        assert "60%" in output  # 3/5 = 60%
        assert "0.85" in output

    def test_coverage_no_children(self) -> None:
        from apps.mcp.formatters import format_coverage_result

        coverage = MockTopicCoverage(
            topic_id=1,
            path="leaf/topic",
            label="Leaf Topic",
            note_count=10,
            child_count=0,
        )

        output = format_coverage_result(coverage)

        assert "Leaf Topic" in output
        assert "Child Topics" not in output


class TestFormatGapsResult:
    """Tests for format_gaps_result."""

    def test_no_gaps(self) -> None:
        from apps.mcp.formatters import format_gaps_result

        output = format_gaps_result([], "math")

        assert "Knowledge Gaps: math" in output
        assert "No gaps found" in output

    def test_with_missing_gaps(self) -> None:
        from apps.mcp.formatters import format_gaps_result

        gaps = [
            MockTopicGap(
                topic_id=1,
                path="math/topology",
                label="Topology",
                description="Study of continuous deformations",
                gap_type="missing",
                note_count=0,
                threshold=1,
            ),
        ]

        output = format_gaps_result(gaps, "math")

        assert "Missing Topics (1)" in output
        assert "Topology" in output
        assert "zero notes" in output.lower()

    def test_with_undercovered_gaps(self) -> None:
        from apps.mcp.formatters import format_gaps_result

        gaps = [
            MockTopicGap(
                topic_id=2,
                path="math/algebra",
                label="Algebra",
                description="Abstract algebra concepts",
                gap_type="undercovered",
                note_count=2,
                threshold=5,
            ),
        ]

        output = format_gaps_result(gaps, "math")

        assert "Undercovered Topics (1)" in output
        assert "Algebra" in output
        assert "2" in output
        assert "5" in output

    def test_mixed_gaps(self) -> None:
        from apps.mcp.formatters import format_gaps_result

        gaps = [
            MockTopicGap(
                topic_id=1,
                path="math/topology",
                label="Topology",
                description=None,
                gap_type="missing",
                note_count=0,
                threshold=1,
            ),
            MockTopicGap(
                topic_id=2,
                path="math/algebra",
                label="Algebra",
                description="Algebra desc",
                gap_type="undercovered",
                note_count=2,
                threshold=5,
            ),
        ]

        output = format_gaps_result(gaps, "math")

        assert "Missing Topics (1)" in output
        assert "Undercovered Topics (1)" in output


class TestFormatDuplicatesResult:
    """Tests for format_duplicates_result."""

    def test_no_duplicates(self) -> None:
        from apps.mcp.formatters import format_duplicates_result

        stats = MockDuplicateStats(notes_scanned=100)
        output = format_duplicates_result([], stats)

        assert "Duplicate Detection Results" in output
        assert "Notes scanned**: 100" in output
        assert "No duplicate clusters found" in output

    def test_with_duplicates(self) -> None:
        from apps.mcp.formatters import format_duplicates_result

        clusters = [
            MockDuplicateCluster(
                representative_id=1,
                representative_text="What is photosynthesis?",
                duplicates=[
                    {
                        "note_id": 2,
                        "similarity": 0.95,
                        "text": "Explain photosynthesis",
                    }
                ],
            )
        ]
        stats = MockDuplicateStats(
            notes_scanned=100,
            clusters_found=1,
            total_duplicates=1,
            avg_cluster_size=2.0,
        )

        output = format_duplicates_result(clusters, stats)

        assert "Clusters found**: 1" in output
        assert "Total duplicates**: 1" in output
        assert "photosynthesis" in output
        assert "95%" in output

    def test_large_cluster(self) -> None:
        from apps.mcp.formatters import format_duplicates_result

        clusters = [
            MockDuplicateCluster(
                representative_id=1,
                representative_text="Main concept",
                duplicates=[
                    {"note_id": i, "similarity": 0.9, "text": f"Duplicate {i}"}
                    for i in range(2, 10)
                ],
            )
        ]
        stats = MockDuplicateStats(clusters_found=1, total_duplicates=8)

        output = format_duplicates_result(clusters, stats)

        assert "...and 3 more" in output  # 8 duplicates, show 5


class TestFormatSyncResult:
    """Tests for format_sync_result."""

    def test_sync_result(self) -> None:
        from apps.mcp.formatters import format_sync_result

        stats = MockSyncStats(
            decks_upserted=10,
            models_upserted=5,
            notes_upserted=1000,
            cards_upserted=2500,
            card_stats_upserted=2500,
            duration_ms=5432,
        )

        output = format_sync_result(stats)

        assert "Sync Complete" in output
        assert "5.4s" in output
        assert "**Decks**: 10" in output
        assert "**Notes**: 1000" in output

    def test_sync_with_deletions(self) -> None:
        from apps.mcp.formatters import format_sync_result

        stats = MockSyncStats(
            notes_upserted=100,
            notes_deleted=5,
            duration_ms=1000,
        )

        output = format_sync_result(stats)

        assert "Notes removed: 5" in output

    def test_sync_no_deletions(self) -> None:
        from apps.mcp.formatters import format_sync_result

        stats = MockSyncStats(
            notes_upserted=100,
            notes_deleted=0,
            duration_ms=1000,
        )

        output = format_sync_result(stats)

        assert "Notes removed" not in output


class TestTruncate:
    """Tests for the _truncate helper."""

    def test_short_text(self) -> None:
        from apps.mcp.formatters import _truncate

        assert _truncate("short", 10) == "short"

    def test_long_text(self) -> None:
        from apps.mcp.formatters import _truncate

        result = _truncate("this is a long text", 10)
        assert result == "this is..."
        assert len(result) == 10

    def test_newlines_replaced(self) -> None:
        from apps.mcp.formatters import _truncate

        result = _truncate("line1\nline2\nline3", 50)
        assert "\n" not in result
        assert result == "line1 line2 line3"


class TestAnkiatlasSync:
    """Tests for ankiatlas_sync tool - only testing file validation."""

    @pytest.mark.asyncio
    async def test_sync_file_not_found(self) -> None:
        from apps.mcp.tools import ankiatlas_sync

        result = await ankiatlas_sync("/nonexistent/collection.anki2")

        assert "**Error**" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_sync_invalid_extension(self) -> None:
        from pathlib import Path
        from unittest.mock import patch

        from apps.mcp.tools import ankiatlas_sync

        with patch.object(Path, "exists", return_value=True):
            result = await ankiatlas_sync("/path/to/file.txt")

        assert "**Error**" in result
        assert ".anki2" in result

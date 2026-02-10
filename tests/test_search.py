"""Tests for hybrid search functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from packages.common.config import Settings
from packages.indexer.embeddings import MockEmbeddingProvider
from packages.search.fts import FTSResult, SearchFilters
from packages.search.fusion import (
    SearchResult,
    reciprocal_rank_fusion,
)
from packages.search.service import SearchService


class TestReciprocalRankFusion:
    """Tests for RRF algorithm."""

    def test_empty_inputs(self) -> None:
        """Test fusion with no results."""
        results, stats = reciprocal_rank_fusion([], [])
        assert results == []
        assert stats.total == 0

    def test_semantic_only(self) -> None:
        """Test fusion with only semantic results."""
        semantic = [(1, 0.95), (2, 0.85), (3, 0.75)]
        results, stats = reciprocal_rank_fusion(semantic, [])

        assert len(results) == 3
        assert stats.semantic_only == 3
        assert stats.fts_only == 0
        assert stats.both == 0

        # First result should be note 1 (highest semantic rank)
        assert results[0].note_id == 1
        assert results[0].semantic_score == 0.95
        assert results[0].fts_score is None

    def test_fts_only(self) -> None:
        """Test fusion with only FTS results."""
        fts: list[tuple[int, float, str | None]] = [(10, 0.8, "headline 10"), (20, 0.6, "headline 20")]
        results, stats = reciprocal_rank_fusion([], fts)

        assert len(results) == 2
        assert stats.semantic_only == 0
        assert stats.fts_only == 2
        assert stats.both == 0

        assert results[0].note_id == 10
        assert results[0].headline == "headline 10"
        assert results[0].fts_score == 0.8

    def test_mixed_results(self) -> None:
        """Test fusion with overlapping results."""
        # Note 1 appears in both, note 2 only in semantic, note 3 only in FTS
        semantic = [(1, 0.9), (2, 0.8)]
        fts: list[tuple[int, float, str | None]] = [(1, 0.7, "hl1"), (3, 0.6, "hl3")]

        results, stats = reciprocal_rank_fusion(semantic, fts)

        assert len(results) == 3
        assert stats.both == 1  # Note 1
        assert stats.semantic_only == 1  # Note 2
        assert stats.fts_only == 1  # Note 3

        # Note 1 should be first (appears in both)
        note1 = next(r for r in results if r.note_id == 1)
        assert note1.semantic_score == 0.9
        assert note1.fts_score == 0.7
        assert note1.headline == "hl1"
        assert "semantic" in note1.sources
        assert "fts" in note1.sources

    def test_limit_results(self) -> None:
        """Test that limit is respected."""
        semantic = [(i, 0.9 - i * 0.1) for i in range(10)]
        results, _ = reciprocal_rank_fusion(semantic, [], limit=5)

        assert len(results) == 5

    def test_weights(self) -> None:
        """Test that weights affect scoring."""
        semantic = [(1, 0.9)]
        fts: list[tuple[int, float, str | None]] = [(2, 0.9, None)]

        # Equal weights: both should have same RRF score
        results, _ = reciprocal_rank_fusion(
            semantic, fts, semantic_weight=1.0, fts_weight=1.0
        )
        assert results[0].rrf_score == results[1].rrf_score

        # Higher semantic weight
        results, _ = reciprocal_rank_fusion(
            semantic, fts, semantic_weight=2.0, fts_weight=1.0
        )
        note1 = next(r for r in results if r.note_id == 1)
        note2 = next(r for r in results if r.note_id == 2)
        assert note1.rrf_score > note2.rrf_score

    def test_rrf_formula(self) -> None:
        """Test RRF score calculation matches formula."""
        # RRF(d) = weight / (k + rank)
        semantic = [(1, 0.9)]  # rank 1
        k = 60

        results, _ = reciprocal_rank_fusion(semantic, [], k=k, semantic_weight=1.0)

        expected_score = 1.0 / (k + 1)  # weight / (k + rank)
        assert abs(results[0].rrf_score - expected_score) < 0.0001


class TestSearchFilters:
    """Tests for SearchFilters dataclass."""

    def test_default_values(self) -> None:
        """Test default filter values."""
        filters = SearchFilters()
        assert filters.deck_names is None
        assert filters.tags is None
        assert filters.model_ids is None
        assert filters.min_ivl is None

    def test_filter_values(self) -> None:
        """Test setting filter values."""
        filters = SearchFilters(
            deck_names=["Deck1", "Deck2"],
            tags=["tag1"],
            min_ivl=21,
            max_lapses=5,
        )
        assert filters.deck_names == ["Deck1", "Deck2"]
        assert filters.tags == ["tag1"]
        assert filters.min_ivl == 21
        assert filters.max_lapses == 5


class TestSearchService:
    """Tests for SearchService."""

    @pytest.fixture
    def settings(self) -> Settings:
        """Create test settings."""
        return Settings(
            embedding_provider="mock",
            embedding_model="mock",
            embedding_dimension=384,
            postgres_url="postgresql://test:test@localhost/test",
            qdrant_url="http://localhost:6333",
        )

    @pytest.fixture
    def mock_embedding_provider(self) -> MockEmbeddingProvider:
        """Create mock embedding provider."""
        return MockEmbeddingProvider(dimension=384)

    @pytest.fixture
    def mock_qdrant_repository(self) -> AsyncMock:
        """Create mock Qdrant repository."""
        repo = AsyncMock()
        repo.search = AsyncMock(return_value=[])
        return repo

    async def test_search_empty_query(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
    ) -> None:
        """Test search with empty query."""
        service = SearchService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        result = await service.search("")
        assert result.results == []
        assert result.stats.total == 0

    async def test_search_semantic_only(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
    ) -> None:
        """Test search with semantic only flag."""
        mock_qdrant_repository.search = AsyncMock(
            return_value=[(1, 0.95), (2, 0.85)]
        )

        service = SearchService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        # Patch FTS search to ensure it's not called
        with patch("packages.search.service.search_fts") as mock_fts:
            result = await service.search("test query", semantic_only=True)

            mock_fts.assert_not_called()
            assert len(result.results) == 2
            assert result.stats.fts_only == 0

    async def test_search_fts_only(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
    ) -> None:
        """Test search with FTS only flag."""
        service = SearchService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        fts_results = [
            FTSResult(note_id=10, rank=0.8, headline="test headline"),
            FTSResult(note_id=20, rank=0.6, headline="another headline"),
        ]

        with patch("packages.search.service.search_fts", return_value=fts_results):
            result = await service.search("test query", fts_only=True)

            assert len(result.results) == 2
            assert result.stats.semantic_only == 0
            # Qdrant should not be called
            mock_qdrant_repository.search.assert_not_called()

    async def test_search_hybrid(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
    ) -> None:
        """Test hybrid search combining both sources."""
        # Semantic results
        mock_qdrant_repository.search = AsyncMock(
            return_value=[(1, 0.95), (2, 0.85)]
        )

        # FTS results (note 1 overlaps, note 3 is new)
        fts_results = [
            FTSResult(note_id=1, rank=0.8, headline="overlap"),
            FTSResult(note_id=3, rank=0.7, headline="fts only"),
        ]

        service = SearchService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        with patch("packages.search.service.search_fts", return_value=fts_results):
            result = await service.search("test query")

            assert len(result.results) == 3  # 1, 2, 3
            assert result.stats.both == 1  # note 1
            assert result.stats.semantic_only == 1  # note 2
            assert result.stats.fts_only == 1  # note 3

    async def test_search_filters_applied(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
    ) -> None:
        """Test that filters are recorded in result."""
        mock_qdrant_repository.search = AsyncMock(return_value=[])

        service = SearchService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        filters = SearchFilters(
            deck_names=["TestDeck"],
            tags=["python"],
            min_ivl=21,
        )

        with patch("packages.search.service.search_fts", return_value=[]):
            result = await service.search("test", filters=filters)

            assert result.filters_applied["deck_names"] == ["TestDeck"]
            assert result.filters_applied["tags"] == ["python"]
            assert result.filters_applied["min_ivl"] == 21


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_sources_both(self) -> None:
        """Test sources when both scores present."""
        result = SearchResult(
            note_id=1,
            rrf_score=0.5,
            semantic_score=0.9,
            fts_score=0.8,
        )
        assert "semantic" in result.sources
        assert "fts" in result.sources

    def test_sources_semantic_only(self) -> None:
        """Test sources with only semantic."""
        result = SearchResult(
            note_id=1,
            rrf_score=0.5,
            semantic_score=0.9,
        )
        assert result.sources == ["semantic"]

    def test_sources_fts_only(self) -> None:
        """Test sources with only FTS."""
        result = SearchResult(
            note_id=1,
            rrf_score=0.5,
            fts_score=0.8,
        )
        assert result.sources == ["fts"]

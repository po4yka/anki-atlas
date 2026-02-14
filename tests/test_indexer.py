"""Tests for indexer service."""

import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import models

from packages.common.config import Settings
from packages.indexer.embeddings import MockEmbeddingProvider
from packages.indexer.qdrant import (
    DimensionMismatchError,
    NotePayload,
    QdrantRepository,
)
from packages.indexer.service import (
    EmbeddingModelChanged,
    IndexService,
    IndexStats,
    NoteForIndexing,
)


@pytest.fixture
def settings() -> Settings:
    """Create test settings."""
    return Settings(
        embedding_provider="mock",
        embedding_model="mock",
        embedding_dimension=384,
        postgres_url="postgresql://test:test@localhost/test",
        qdrant_url="http://localhost:6333",
    )


@pytest.fixture
def mock_embedding_provider() -> MockEmbeddingProvider:
    """Create mock embedding provider."""
    return MockEmbeddingProvider(dimension=384)


@pytest.fixture
def mock_qdrant_repository() -> AsyncMock:
    """Create mock Qdrant repository."""
    repo = AsyncMock(spec=QdrantRepository)
    repo.ensure_collection = AsyncMock(return_value=False)
    repo.get_existing_hashes = AsyncMock(return_value={})
    repo.upsert_vectors = AsyncMock(return_value=0)
    repo.delete_vectors = AsyncMock(return_value=0)
    return repo


@pytest.fixture
def sample_notes() -> list[NoteForIndexing]:
    """Create sample notes for testing."""
    return [
        NoteForIndexing(
            note_id=1,
            model_id=100,
            normalized_text="What is Python? A programming language.",
            tags=["python", "programming"],
            deck_names=["Programming::Python"],
            mature=True,
            lapses=2,
            reps=15,
            fail_rate=0.1,
        ),
        NoteForIndexing(
            note_id=2,
            model_id=100,
            normalized_text="What is a variable? A named storage location.",
            tags=["basics"],
            deck_names=["Programming::Basics"],
            mature=False,
            lapses=0,
            reps=3,
            fail_rate=None,
        ),
    ]


class TestIndexService:
    """Tests for IndexService."""

    async def test_index_notes_empty(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
    ) -> None:
        """Test indexing empty list of notes."""
        service = IndexService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        stats = await service.index_notes([])
        assert stats.notes_processed == 0
        assert stats.notes_embedded == 0
        assert stats.notes_skipped == 0
        assert stats.errors == []

    async def test_index_notes_success(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
        sample_notes: list[NoteForIndexing],
    ) -> None:
        """Test successful indexing of notes."""
        mock_qdrant_repository.upsert_vectors = AsyncMock(return_value=2)

        service = IndexService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        stats = await service.index_notes(sample_notes)

        assert stats.notes_processed == 2
        assert stats.notes_embedded == 2
        assert stats.notes_skipped == 0
        assert stats.errors == []

        # Verify collection was ensured
        mock_qdrant_repository.ensure_collection.assert_called_once_with(384)

        # Verify vectors were upserted
        mock_qdrant_repository.upsert_vectors.assert_called_once()
        call_args = mock_qdrant_repository.upsert_vectors.call_args
        vectors, payloads = call_args.args
        assert len(vectors) == 2
        assert len(payloads) == 2
        assert all(len(v) == 384 for v in vectors)

    async def test_index_notes_skip_unchanged(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
        sample_notes: list[NoteForIndexing],
    ) -> None:
        """Test that unchanged notes are skipped."""
        # Simulate note 1 already having same hash
        existing_hash = mock_embedding_provider.content_hash(sample_notes[0].normalized_text)
        mock_qdrant_repository.get_existing_hashes = AsyncMock(return_value={1: existing_hash})
        mock_qdrant_repository.upsert_vectors = AsyncMock(return_value=1)

        service = IndexService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        stats = await service.index_notes(sample_notes)

        assert stats.notes_processed == 2
        assert stats.notes_embedded == 1
        assert stats.notes_skipped == 1

    async def test_index_notes_force_reindex(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
        sample_notes: list[NoteForIndexing],
    ) -> None:
        """Test force reindex ignores existing hashes."""
        # Simulate all notes having same hash
        existing_hash = mock_embedding_provider.content_hash(sample_notes[0].normalized_text)
        mock_qdrant_repository.get_existing_hashes = AsyncMock(
            return_value={1: existing_hash, 2: existing_hash}
        )
        mock_qdrant_repository.upsert_vectors = AsyncMock(return_value=2)

        service = IndexService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        stats = await service.index_notes(sample_notes, force_reindex=True)

        assert stats.notes_processed == 2
        assert stats.notes_embedded == 2
        assert stats.notes_skipped == 0

        # get_existing_hashes should not be called when force_reindex=True
        mock_qdrant_repository.get_existing_hashes.assert_not_called()

    async def test_index_notes_embedding_error(
        self,
        settings: Settings,
        mock_qdrant_repository: AsyncMock,
        sample_notes: list[NoteForIndexing],
    ) -> None:
        """Test handling of embedding errors."""
        # Create provider that raises an error
        failing_provider = MagicMock(spec=MockEmbeddingProvider)
        failing_provider.content_hash = lambda _: "hash"
        failing_provider.dimension = 384
        failing_provider.embed = AsyncMock(side_effect=RuntimeError("API error"))

        service = IndexService(
            settings=settings,
            embedding_provider=failing_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        stats = await service.index_notes(sample_notes)

        assert stats.notes_processed == 2
        assert stats.notes_embedded == 0
        assert len(stats.errors) == 1
        assert "Embedding failed" in stats.errors[0]

    async def test_delete_notes(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
    ) -> None:
        """Test deleting notes from index."""
        mock_qdrant_repository.delete_vectors = AsyncMock(return_value=3)

        service = IndexService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        deleted = await service.delete_notes([1, 2, 3])
        assert deleted == 3
        mock_qdrant_repository.delete_vectors.assert_called_once_with([1, 2, 3])

    async def test_delete_notes_empty(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
    ) -> None:
        """Test deleting empty list returns zero."""
        service = IndexService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        deleted = await service.delete_notes([])
        assert deleted == 0
        mock_qdrant_repository.delete_vectors.assert_not_called()

    async def test_model_change_detected_raises(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
    ) -> None:
        """Test that a changed embedding version raises EmbeddingModelChanged."""
        service = IndexService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        stored_version = "1:openai/text-embedding-3-small:1536"
        with patch.object(
            service,
            "_get_stored_embedding_version",
            new_callable=AsyncMock,
            return_value=stored_version,
        ):
            with pytest.raises(EmbeddingModelChanged) as exc_info:
                await service.index_from_database(force_reindex=False)

            assert exc_info.value.stored_version == stored_version
            assert "mock/test" in exc_info.value.current_version

    async def test_model_change_with_force_reindex_recreates(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
    ) -> None:
        """Test force_reindex recreates collection on model change."""
        service = IndexService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )
        mock_qdrant_repository.recreate_collection = AsyncMock()

        stored_version = "1:openai/text-embedding-3-small:1536"
        with (
            patch.object(
                service,
                "_get_stored_embedding_version",
                new_callable=AsyncMock,
                return_value=stored_version,
            ),
            patch(
                "packages.indexer.service.get_connection",
            ) as mock_conn,
        ):
            # Simulate empty DB (0 notes) so index_from_database finishes quickly
            mock_result = AsyncMock()
            mock_result.fetchone = AsyncMock(return_value={"count": 0})
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.execute = AsyncMock(return_value=mock_result)
            mock_ctx.commit = AsyncMock()
            mock_conn.return_value = mock_ctx

            await service.index_from_database(force_reindex=True)

            mock_qdrant_repository.recreate_collection.assert_called_once_with(384)

    async def test_first_run_no_stored_version_proceeds(
        self,
        settings: Settings,
        mock_embedding_provider: MockEmbeddingProvider,
        mock_qdrant_repository: AsyncMock,
    ) -> None:
        """Test first run with no stored version proceeds normally."""
        service = IndexService(
            settings=settings,
            embedding_provider=mock_embedding_provider,
            qdrant_repository=mock_qdrant_repository,
        )

        with (
            patch.object(
                service,
                "_get_stored_embedding_version",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "packages.indexer.service.get_connection",
            ) as mock_conn,
        ):
            mock_result = AsyncMock()
            mock_result.fetchone = AsyncMock(return_value={"count": 0})
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.execute = AsyncMock(return_value=mock_result)
            mock_ctx.commit = AsyncMock()
            mock_conn.return_value = mock_ctx

            # Should NOT raise
            stats = await service.index_from_database(force_reindex=False)
            assert stats.notes_processed == 0


class TestDimensionMismatch:
    """Tests for Qdrant dimension validation."""

    async def test_dimension_mismatch_in_ensure_collection(self) -> None:
        """Test that mismatched dimensions raise DimensionMismatchError."""
        repo = QdrantRepository(
            settings=Settings(
                embedding_provider="mock",
                embedding_model="mock",
                embedding_dimension=384,
                postgres_url="postgresql://test:test@localhost/test",
                qdrant_url="http://localhost:6333",
            ),
        )

        mock_client = AsyncMock()
        # Collection exists
        mock_collection = MagicMock()
        mock_collection.name = "anki_notes"
        mock_client.get_collections = AsyncMock(
            return_value=MagicMock(collections=[mock_collection])
        )
        # Collection has dimension 384
        mock_info = MagicMock()
        mock_info.config.params.vectors = models.VectorParams(
            size=384, distance=models.Distance.COSINE
        )
        mock_client.get_collection = AsyncMock(return_value=mock_info)

        repo._client = mock_client

        with pytest.raises(DimensionMismatchError) as exc_info:
            await repo.ensure_collection(1536)

        assert exc_info.value.expected == 1536
        assert exc_info.value.actual == 384


class TestSparseEncoding:
    """Tests for sparse vector encoding."""

    def test_text_to_sparse_vector_empty(self) -> None:
        """Empty text should produce an empty sparse vector."""
        vector = QdrantRepository.text_to_sparse_vector("")
        assert vector.indices == []
        assert vector.values == []

    def test_text_to_sparse_vector_is_normalized_and_deduplicated(self) -> None:
        """Repeated terms should be merged and L2-normalized."""
        vector = QdrantRepository.text_to_sparse_vector("Python python java")

        assert len(vector.indices) == 2
        assert vector.indices == sorted(vector.indices)

        l2_norm = math.sqrt(sum(v * v for v in vector.values))
        assert abs(l2_norm - 1.0) < 1e-6


class TestNoteForIndexing:
    """Tests for NoteForIndexing dataclass."""

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        note = NoteForIndexing(
            note_id=1,
            model_id=100,
            normalized_text="test",
            tags=[],
            deck_names=[],
        )
        assert note.mature is False
        assert note.lapses == 0
        assert note.reps == 0
        assert note.fail_rate is None


class TestIndexStats:
    """Tests for IndexStats dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        stats = IndexStats()
        assert stats.notes_processed == 0
        assert stats.notes_embedded == 0
        assert stats.notes_skipped == 0
        assert stats.notes_deleted == 0
        assert stats.errors == []


class TestNotePayload:
    """Tests for NotePayload dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        payload = NotePayload(
            note_id=1,
            deck_names=["Deck::Subdeck"],
            tags=["tag1", "tag2"],
            model_id=100,
            content_hash="abc123",
            mature=True,
            lapses=5,
            reps=20,
            fail_rate=0.15,
        )
        d = payload.to_dict()
        assert d["note_id"] == 1
        assert d["deck_names"] == ["Deck::Subdeck"]
        assert d["tags"] == ["tag1", "tag2"]
        assert d["model_id"] == 100
        assert d["content_hash"] == "abc123"
        assert d["mature"] is True
        assert d["lapses"] == 5
        assert d["reps"] == 20
        assert d["fail_rate"] == 0.15

    def test_default_values(self) -> None:
        """Test default values."""
        payload = NotePayload(
            note_id=1,
            deck_names=[],
            tags=[],
            model_id=100,
            content_hash="abc",
        )
        assert payload.mature is False
        assert payload.lapses == 0
        assert payload.reps == 0
        assert payload.fail_rate is None

"""Tests for indexer service."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from packages.common.config import Settings
from packages.indexer.embeddings import MockEmbeddingProvider
from packages.indexer.qdrant import NotePayload, QdrantRepository
from packages.indexer.service import IndexService, IndexStats, NoteForIndexing


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
        existing_hash = mock_embedding_provider.content_hash(
            sample_notes[0].normalized_text
        )
        mock_qdrant_repository.get_existing_hashes = AsyncMock(
            return_value={1: existing_hash}
        )
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
        existing_hash = mock_embedding_provider.content_hash(
            sample_notes[0].normalized_text
        )
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

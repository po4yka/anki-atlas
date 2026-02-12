"""Tests for embedding providers."""

import pytest

from packages.common.config import Settings
from packages.indexer.embeddings import (
    MockEmbeddingProvider,
    get_embedding_provider,
)


class TestMockEmbeddingProvider:
    """Tests for MockEmbeddingProvider."""

    @pytest.fixture
    def provider(self) -> MockEmbeddingProvider:
        """Create mock provider."""
        return MockEmbeddingProvider(dimension=384)

    def test_model_name(self, provider: MockEmbeddingProvider) -> None:
        """Test model name property."""
        assert provider.model_name == "mock/test"

    def test_dimension(self, provider: MockEmbeddingProvider) -> None:
        """Test dimension property."""
        assert provider.dimension == 384

    async def test_embed_single(self, provider: MockEmbeddingProvider) -> None:
        """Test embedding a single text."""
        result = await provider.embed_single("test text")
        assert len(result) == 384
        assert all(isinstance(v, float) for v in result)
        assert all(-1.0 <= v <= 1.0 for v in result)

    async def test_embed_multiple(self, provider: MockEmbeddingProvider) -> None:
        """Test embedding multiple texts."""
        texts = ["hello world", "foo bar", "test"]
        results = await provider.embed(texts)
        assert len(results) == 3
        for embedding in results:
            assert len(embedding) == 384

    async def test_embed_empty(self, provider: MockEmbeddingProvider) -> None:
        """Test embedding empty list."""
        results = await provider.embed([])
        assert results == []

    async def test_embed_deterministic(self, provider: MockEmbeddingProvider) -> None:
        """Test that embeddings are deterministic for same input."""
        text = "deterministic test"
        result1 = await provider.embed_single(text)
        result2 = await provider.embed_single(text)
        assert result1 == result2

    async def test_embed_different_texts_different_vectors(
        self, provider: MockEmbeddingProvider
    ) -> None:
        """Test that different texts produce different embeddings."""
        result1 = await provider.embed_single("text one")
        result2 = await provider.embed_single("text two")
        assert result1 != result2

    def test_content_hash(self, provider: MockEmbeddingProvider) -> None:
        """Test content hash generation."""
        hash1 = provider.content_hash("test text")
        hash2 = provider.content_hash("test text")
        hash3 = provider.content_hash("different text")

        assert hash1 == hash2  # Same text, same hash
        assert hash1 != hash3  # Different text, different hash
        assert len(hash1) == 16  # Hash length

    def test_content_hash_includes_model(self) -> None:
        """Test that content hash includes model name."""
        provider1 = MockEmbeddingProvider(dimension=384)

        # Create a subclass with different model name
        class DifferentMock(MockEmbeddingProvider):
            @property
            def model_name(self) -> str:
                return "mock/different"

        provider2 = DifferentMock(dimension=384)

        hash1 = provider1.content_hash("same text")
        hash2 = provider2.content_hash("same text")
        assert hash1 != hash2  # Different model, different hash


class TestGetEmbeddingProvider:
    """Tests for get_embedding_provider factory."""

    def test_get_mock_provider(self) -> None:
        """Test getting mock provider from settings."""
        settings = Settings(
            embedding_provider="mock",
            embedding_model="mock",
            embedding_dimension=256,
            postgres_url="postgresql://test:test@localhost/test",
            qdrant_url="http://localhost:6333",
        )
        provider = get_embedding_provider(settings)
        assert isinstance(provider, MockEmbeddingProvider)
        assert provider.dimension == 256

    def test_unknown_provider_raises(self) -> None:
        """Test that unknown provider raises ValueError."""
        settings = Settings(
            embedding_provider="unknown",
            embedding_model="test",
            embedding_dimension=384,
            postgres_url="postgresql://test:test@localhost/test",
            qdrant_url="http://localhost:6333",
        )
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            get_embedding_provider(settings)

"""Embedding providers for text vectorization."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from packages.common.config import Settings, get_settings

if TYPE_CHECKING:
    import openai
    from sentence_transformers import SentenceTransformer


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for version tracking."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        results = await self.embed([text])
        return results[0]

    def content_hash(self, text: str) -> str:
        """Compute content hash for change detection.

        Includes model name so hash changes when model changes.
        """
        content = f"{self.model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        batch_size: int = 100,
    ) -> None:
        """Initialize OpenAI embedding provider.

        Args:
            model: OpenAI model name.
            dimension: Embedding dimension (can be reduced for some models).
            batch_size: Maximum texts per API call.
        """
        self._model = model
        self._dimension = dimension
        self._batch_size = batch_size
        self._client: openai.AsyncOpenAI | None = None  # Lazy initialized

    @property
    def model_name(self) -> str:
        return f"openai/{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self) -> Any:
        """Lazily initialize OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "OpenAI package not installed. Install with: uv sync --extra embeddings-openai"
                ) from e
            self._client = openai.AsyncOpenAI()
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenAI API."""
        if not texts:
            return []

        client = self._get_client()
        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            response = await client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimension,
            )

            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            batch_embeddings = [item.embedding for item in sorted_data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
    ) -> None:
        """Initialize local embedding provider.

        Args:
            model: HuggingFace model name.
            batch_size: Maximum texts per batch.
        """
        self._model_name = model
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None  # Lazy initialized
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return f"local/{self._model_name}"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Load model to get dimension
            model = self._get_model()
            dim = model.get_sentence_embedding_dimension()
            self._dimension = int(dim) if dim else 384
        return self._dimension

    def _get_model(self) -> Any:
        """Lazily initialize sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: uv sync --extra embeddings-local"
                ) from e
            self._model = SentenceTransformer(self._model_name)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using local model.

        Note: This runs synchronously but is wrapped in async for interface consistency.
        For production, consider using run_in_executor.
        """
        if not texts:
            return []

        import asyncio

        model = self._get_model()

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
        )

        return [emb.tolist() for emb in embeddings]


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 384) -> None:
        """Initialize mock provider."""
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return "mock/test"

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic fake embeddings based on text hash."""
        import struct

        embeddings: list[list[float]] = []
        for text in texts:
            # Use hash to generate deterministic "embedding"
            text_hash = hashlib.md5(text.encode()).digest()
            # Repeat hash to fill dimension
            repeated = (text_hash * ((self._dimension // 16) + 1))[: self._dimension]
            # Convert bytes to floats in [-1, 1]
            embedding = [
                (struct.unpack("B", bytes([b]))[0] / 127.5) - 1.0 for b in repeated
            ]
            embeddings.append(embedding)
        return embeddings


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Factory function to create embedding provider from settings.

    Args:
        settings: Application settings. If None, uses default settings.

    Returns:
        Configured embedding provider.
    """
    if settings is None:
        settings = get_settings()

    provider_type = settings.embedding_provider.lower()

    if provider_type == "openai":
        return OpenAIEmbeddingProvider(
            model=settings.embedding_model,
            dimension=settings.embedding_dimension,
        )
    elif provider_type == "local":
        return LocalEmbeddingProvider(
            model=settings.embedding_model,
        )
    elif provider_type == "mock":
        return MockEmbeddingProvider(
            dimension=settings.embedding_dimension,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")

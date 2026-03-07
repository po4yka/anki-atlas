"""Base class for services that depend on embedding provider and Qdrant."""

from __future__ import annotations

from packages.common.config import Settings, get_settings
from packages.indexer.embeddings import EmbeddingProvider, get_embedding_provider
from packages.indexer.qdrant import QdrantRepository, get_qdrant_repository


class ServiceBase:
    """Shared initialization and lazy accessors for embedding + Qdrant services."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        qdrant_repository: QdrantRepository | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embedding_provider = embedding_provider
        self._qdrant_repository = qdrant_repository

    async def get_embedding_provider(self) -> EmbeddingProvider:
        """Get or create embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider(self.settings)
        return self._embedding_provider

    async def get_qdrant_repository(self) -> QdrantRepository:
        """Get or create Qdrant repository."""
        if self._qdrant_repository is None:
            self._qdrant_repository = await get_qdrant_repository(self.settings)
        return self._qdrant_repository

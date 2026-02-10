# Embedding, qdrant upsert, incremental sync

from packages.indexer.embeddings import (
    EmbeddingProvider,
    LocalEmbeddingProvider,
    MockEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)
from packages.indexer.qdrant import (
    NotePayload,
    QdrantRepository,
    get_qdrant_repository,
)
from packages.indexer.service import (
    IndexService,
    IndexStats,
    NoteForIndexing,
    index_all_notes,
)

__all__ = [
    "EmbeddingProvider",
    "IndexService",
    "IndexStats",
    "LocalEmbeddingProvider",
    "MockEmbeddingProvider",
    "NoteForIndexing",
    "NotePayload",
    "OpenAIEmbeddingProvider",
    "QdrantRepository",
    "get_embedding_provider",
    "get_qdrant_repository",
    "index_all_notes",
]

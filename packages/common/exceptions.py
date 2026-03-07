"""Custom exception hierarchy for Anki Atlas."""

from __future__ import annotations


class AnkiAtlasError(Exception):
    """Base exception for all application errors."""

    def __init__(self, message: str, *, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


class DatabaseError(AnkiAtlasError):
    """Base class for database-related errors."""


class DatabaseConnectionError(DatabaseError):
    """Database connection failed or was lost."""


class MigrationError(DatabaseError):
    """Database migration failed."""


class VectorStoreError(AnkiAtlasError):
    """Base class for vector database (Qdrant) errors."""


class VectorStoreConnectionError(VectorStoreError):
    """Cannot connect to Qdrant server."""


class CollectionError(VectorStoreError):
    """Collection operation failed (create, delete, query)."""


class EmbeddingError(AnkiAtlasError):
    """Base class for embedding generation errors."""


class EmbeddingAPIError(EmbeddingError):
    """Embedding API call failed (network, auth, rate limit)."""


class EmbeddingTimeoutError(EmbeddingError):
    """Embedding API timed out."""


class SyncError(AnkiAtlasError):
    """Anki collection sync failed."""


class CollectionNotFoundError(SyncError):
    """Anki collection file not found."""


class ConfigurationError(AnkiAtlasError):
    """Invalid or missing configuration."""


class NotFoundError(AnkiAtlasError):
    """Requested resource not found."""


class ConflictError(AnkiAtlasError):
    """Operation conflicts with current state (e.g., dimension mismatch)."""


class EmbeddingModelChanged(ConflictError):
    """Raised when embedding model changed since last indexing."""

    def __init__(self, stored: str, current: str) -> None:
        self.stored_version = stored
        self.current_version = current
        super().__init__(
            f"Embedding model changed: '{stored}' -> '{current}'. "
            f"Use --force-reindex to re-embed all notes with the new model."
        )


class CardGenerationError(AnkiAtlasError):
    """Card generation failed."""


class CardValidationError(AnkiAtlasError):
    """Card validation failed."""


class ProviderError(AnkiAtlasError):
    """LLM provider operation failed."""


class ObsidianParseError(AnkiAtlasError):
    """Obsidian note parsing failed."""


class SyncConflictError(AnkiAtlasError):
    """Sync conflict between local and remote state."""


class AnkiConnectError(AnkiAtlasError):
    """AnkiConnect communication failed."""


class AnkiReaderError(AnkiAtlasError):
    """Error reading Anki collection."""


class DimensionMismatchError(VectorStoreError):
    """Raised when requested dimension doesn't match existing collection."""

    def __init__(self, collection: str, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Collection '{collection}' has dimension {actual}, "
            f"but provider requires {expected}. "
            f"Use --force-reindex to recreate the collection."
        )


class JobBackendUnavailableError(AnkiAtlasError):
    """Raised when Redis/arq backend is unavailable."""

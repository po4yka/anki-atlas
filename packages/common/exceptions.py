"""Custom exception hierarchy for Anki Atlas.

This module defines application-specific exceptions that provide:
- Clear error categorization for debugging
- Consistent HTTP status code mapping in API
- Structured logging context
"""

from __future__ import annotations


class AnkiAtlasError(Exception):
    """Base exception for all application errors.

    All custom exceptions inherit from this to enable:
    - Centralized exception handling in API middleware
    - Consistent error logging patterns
    - Type-safe error catching
    """

    def __init__(self, message: str, *, context: dict[str, object] | None = None) -> None:
        """Initialize error with message and optional context.

        Args:
            message: Human-readable error description.
            context: Additional key-value pairs for structured logging.
        """
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

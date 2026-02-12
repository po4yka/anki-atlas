"""Tests for custom exception hierarchy."""

import pytest

from packages.common.exceptions import (
    AnkiAtlasError,
    CollectionError,
    CollectionNotFoundError,
    ConfigurationError,
    ConflictError,
    DatabaseConnectionError,
    DatabaseError,
    EmbeddingAPIError,
    EmbeddingError,
    EmbeddingTimeoutError,
    MigrationError,
    NotFoundError,
    SyncError,
    VectorStoreConnectionError,
    VectorStoreError,
)


class TestAnkiAtlasError:
    """Tests for base exception class."""

    def test_basic_message(self) -> None:
        """Test exception with just a message."""
        err = AnkiAtlasError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.context == {}

    def test_with_context(self) -> None:
        """Test exception with context dict."""
        err = AnkiAtlasError(
            "Operation failed",
            context={"note_id": 123, "deck": "Default"},
        )
        assert str(err) == "Operation failed"
        assert err.context == {"note_id": 123, "deck": "Default"}

    def test_empty_context(self) -> None:
        """Test context defaults to empty dict."""
        err = AnkiAtlasError("Error")
        assert isinstance(err.context, dict)


class TestExceptionHierarchy:
    """Tests for exception inheritance."""

    def test_database_errors_inherit_from_base(self) -> None:
        """Database errors should inherit from AnkiAtlasError."""
        assert issubclass(DatabaseError, AnkiAtlasError)
        assert issubclass(DatabaseConnectionError, DatabaseError)
        assert issubclass(MigrationError, DatabaseError)

    def test_vector_store_errors_inherit_from_base(self) -> None:
        """Vector store errors should inherit from AnkiAtlasError."""
        assert issubclass(VectorStoreError, AnkiAtlasError)
        assert issubclass(VectorStoreConnectionError, VectorStoreError)
        assert issubclass(CollectionError, VectorStoreError)

    def test_embedding_errors_inherit_from_base(self) -> None:
        """Embedding errors should inherit from AnkiAtlasError."""
        assert issubclass(EmbeddingError, AnkiAtlasError)
        assert issubclass(EmbeddingAPIError, EmbeddingError)
        assert issubclass(EmbeddingTimeoutError, EmbeddingError)

    def test_sync_errors_inherit_from_base(self) -> None:
        """Sync errors should inherit from AnkiAtlasError."""
        assert issubclass(SyncError, AnkiAtlasError)
        assert issubclass(CollectionNotFoundError, SyncError)

    def test_other_errors_inherit_from_base(self) -> None:
        """Other errors should inherit from AnkiAtlasError."""
        assert issubclass(ConfigurationError, AnkiAtlasError)
        assert issubclass(NotFoundError, AnkiAtlasError)
        assert issubclass(ConflictError, AnkiAtlasError)


class TestExceptionCatching:
    """Tests for catching exceptions in various ways."""

    def test_catch_by_base_class(self) -> None:
        """Should be able to catch all custom exceptions with base class."""
        errors = [
            DatabaseConnectionError("db error"),
            VectorStoreConnectionError("qdrant error"),
            EmbeddingAPIError("openai error"),
            ConfigurationError("config error"),
        ]

        for err in errors:
            with pytest.raises(AnkiAtlasError):
                raise err

    def test_catch_database_group(self) -> None:
        """Should be able to catch all database errors with DatabaseError."""
        with pytest.raises(DatabaseError):
            raise DatabaseConnectionError("connection failed")

        with pytest.raises(DatabaseError):
            raise MigrationError("migration failed")

    def test_catch_vector_store_group(self) -> None:
        """Should be able to catch all vector errors with VectorStoreError."""
        with pytest.raises(VectorStoreError):
            raise VectorStoreConnectionError("connection failed")

        with pytest.raises(VectorStoreError):
            raise CollectionError("collection error")

    def test_catch_specific_exception(self) -> None:
        """Should be able to catch specific exceptions."""
        with pytest.raises(DatabaseConnectionError):
            raise DatabaseConnectionError("connection failed")

        # Should not catch wrong type
        with pytest.raises(VectorStoreConnectionError):
            raise VectorStoreConnectionError("qdrant failed")

"""Tests for configuration validation."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from packages.common.config import Settings


class TestEmbeddingDimensionValidation:
    """Tests for embedding_dimension validator."""

    def test_valid_dimensions_accepted(self) -> None:
        """Test that valid embedding dimensions are accepted."""
        valid_dims = [384, 768, 1024, 1536, 3072]
        for dim in valid_dims:
            settings = Settings(embedding_dimension=dim)
            assert settings.embedding_dimension == dim

    def test_invalid_dimension_rejected(self) -> None:
        """Test that invalid embedding dimensions raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(embedding_dimension=512)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "Unsupported embedding dimension" in str(errors[0]["msg"])

    def test_default_dimension_is_valid(self) -> None:
        """Test that default dimension (1536) is valid."""
        settings = Settings()
        assert settings.embedding_dimension == 1536

    def test_mock_provider_allows_custom_dimension(self) -> None:
        """Test that mock provider allows non-standard positive dimensions."""
        settings = Settings(
            embedding_provider="mock",
            embedding_dimension=256,
        )
        assert settings.embedding_dimension == 256

    def test_non_positive_dimension_rejected(self) -> None:
        """Test that non-positive dimensions are rejected for all providers."""
        with pytest.raises(ValidationError):
            Settings(
                embedding_provider="mock",
                embedding_dimension=0,
            )


class TestPostgresUrlValidation:
    """Tests for postgres_url validator."""

    def test_postgresql_scheme_accepted(self) -> None:
        """Test that postgresql:// scheme is accepted."""
        settings = Settings(postgres_url="postgresql://user:pass@localhost/db")
        assert settings.postgres_url.startswith("postgresql://")

    def test_postgres_scheme_accepted(self) -> None:
        """Test that postgres:// scheme is accepted."""
        settings = Settings(postgres_url="postgres://user:pass@localhost/db")
        assert settings.postgres_url.startswith("postgres://")

    def test_invalid_scheme_rejected(self) -> None:
        """Test that invalid schemes raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(postgres_url="mysql://user:pass@localhost/db")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "PostgreSQL connection string" in str(errors[0]["msg"])

    def test_default_url_is_valid(self) -> None:
        """Test that default postgres URL is valid."""
        settings = Settings()
        assert settings.postgres_url.startswith("postgresql://")


class TestQdrantUrlValidation:
    """Tests for qdrant_url validator."""

    def test_http_scheme_accepted(self) -> None:
        """Test that http:// scheme is accepted."""
        settings = Settings(qdrant_url="http://localhost:6333")
        assert settings.qdrant_url.startswith("http://")

    def test_https_scheme_accepted(self) -> None:
        """Test that https:// scheme is accepted."""
        settings = Settings(qdrant_url="https://qdrant.example.com:6333")
        assert settings.qdrant_url.startswith("https://")

    def test_invalid_scheme_rejected(self) -> None:
        """Test that invalid schemes raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(qdrant_url="grpc://localhost:6334")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "HTTP(S) URL" in str(errors[0]["msg"])

    def test_default_url_is_valid(self) -> None:
        """Test that default Qdrant URL is valid."""
        settings = Settings()
        assert settings.qdrant_url.startswith("http")


class TestQdrantQuantizationSettings:
    """Tests for Qdrant quantization settings."""

    def test_valid_quantization_modes(self) -> None:
        """Test that valid quantization modes are accepted."""
        for mode in ["none", "scalar", "binary"]:
            settings = Settings(qdrant_quantization=mode)  # type: ignore[arg-type]
            assert settings.qdrant_quantization == mode

    def test_invalid_quantization_mode_rejected(self) -> None:
        """Test that invalid quantization mode raises ValidationError."""
        with pytest.raises(ValidationError):
            Settings(qdrant_quantization="invalid")  # type: ignore[arg-type]

    def test_default_quantization_is_scalar(self) -> None:
        """Test that default quantization is scalar."""
        settings = Settings()
        assert settings.qdrant_quantization == "scalar"


class TestQdrantOnDiskSettings:
    """Tests for Qdrant on-disk storage settings."""

    def test_on_disk_true(self) -> None:
        """Test that on_disk=True is accepted."""
        settings = Settings(qdrant_on_disk=True)
        assert settings.qdrant_on_disk is True

    def test_on_disk_false(self) -> None:
        """Test that on_disk=False is accepted."""
        settings = Settings(qdrant_on_disk=False)
        assert settings.qdrant_on_disk is False

    def test_default_is_false(self) -> None:
        """Test that default on_disk is False."""
        settings = Settings()
        assert settings.qdrant_on_disk is False


class TestEnvironmentVariables:
    """Tests for loading settings from environment variables."""

    def test_loads_from_env_with_prefix(self) -> None:
        """Test that settings load from ANKIATLAS_ prefixed env vars."""
        with patch.dict(os.environ, {"ANKIATLAS_DEBUG": "true"}):
            # Need to bypass cache for this test
            settings = Settings()
            assert settings.debug is True

    def test_debug_default_is_false(self) -> None:
        """Test that default debug is False."""
        settings = Settings()
        assert settings.debug is False

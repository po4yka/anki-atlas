"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="ANKIATLAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    postgres_url: str = Field(
        default="postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas",
        description="PostgreSQL connection URL",
    )

    # Vector store
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    qdrant_quantization: Literal["none", "scalar", "binary"] = Field(
        default="scalar",
        description="Qdrant quantization mode for memory optimization",
    )
    qdrant_on_disk: bool = Field(
        default=False,
        description="Store vectors on disk for very large collections",
    )

    # Embedding configuration
    embedding_provider: str = Field(
        default="openai",
        description="Embedding provider: 'openai' or 'local'",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension",
    )

    # API configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")

    # Anki source
    anki_collection_path: str | None = Field(
        default=None,
        description="Path to Anki collection.anki2 file",
    )

    @field_validator("embedding_dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        """Validate embedding dimension is a known model dimension."""
        valid_dimensions = {384, 768, 1024, 1536, 3072}
        if v not in valid_dimensions:
            raise ValueError(
                f"Unsupported embedding dimension: {v}. "
                f"Valid dimensions: {sorted(valid_dimensions)}"
            )
        return v

    @field_validator("postgres_url")
    @classmethod
    def validate_postgres_url(cls, v: str) -> str:
        """Validate PostgreSQL connection URL format."""
        if not v.startswith(("postgresql://", "postgres://")):
            raise ValueError("postgres_url must be a PostgreSQL connection string")
        return v

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        """Validate Qdrant URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("qdrant_url must be an HTTP(S) URL")
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator
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

    # Async jobs
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for async job queue",
    )
    job_queue_name: str = Field(
        default="ankiatlas_jobs",
        description="Arq queue name for background jobs",
    )
    job_result_ttl_seconds: int = Field(
        default=86400,
        description="How long to keep job metadata/results in Redis",
    )
    job_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for failed background jobs",
    )

    # Embedding configuration
    embedding_provider: str = Field(
        default="openai",
        description="Embedding provider: 'openai', 'google', 'local', or 'mock'",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding vector dimension",
    )
    rerank_enabled: bool = Field(
        default=False,
        description="Enable second-stage reranking with a cross-encoder model",
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Sentence-transformers CrossEncoder model for reranking",
    )
    rerank_top_n: int = Field(
        default=50,
        description="Number of hybrid candidates to rerank",
    )
    rerank_batch_size: int = Field(
        default=32,
        description="Batch size for reranking model inference",
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
    def validate_dimension(cls, v: int, info: ValidationInfo) -> int:
        """Validate embedding dimension with provider-specific rules."""
        if v <= 0:
            raise ValueError("embedding_dimension must be positive")

        provider = str(info.data.get("embedding_provider", "openai")).lower()
        if provider == "mock":
            return v

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

    @field_validator("redis_url")
    @classmethod
    def validate_redis_url(cls, v: str) -> str:
        """Validate Redis URL format."""
        if not v.startswith(("redis://", "rediss://")):
            raise ValueError("redis_url must be a Redis URL")
        return v

    @field_validator("job_result_ttl_seconds", "job_max_retries", "rerank_top_n", "rerank_batch_size")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        """Validate integer settings that must be positive."""
        if v <= 0:
            raise ValueError("value must be positive")
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic import Field
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


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

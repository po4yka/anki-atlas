"""Anki Atlas API - FastAPI application."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from packages.common.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    settings = get_settings()
    app.state.settings = settings
    yield
    # Shutdown


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Anki Atlas",
        description="Searchable hybrid index for Anki collections with agent-friendly tools",
        version="0.1.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    @app.get("/health", response_class=JSONResponse)
    async def health() -> dict[str, Any]:
        """Health check endpoint.

        Returns basic status and configuration info.
        Does not check database connectivity (use /ready for that).
        """
        return {
            "status": "healthy",
            "version": "0.1.0",
            "services": {
                "postgres": {"configured": True},
                "qdrant": {"configured": True},
            },
        }

    @app.get("/ready", response_class=JSONResponse)
    async def ready() -> dict[str, Any]:
        """Readiness check - verifies all dependencies are available.

        TODO: Implement actual connectivity checks for postgres and qdrant.
        """
        return {
            "status": "ready",
            "checks": {
                "postgres": "not_implemented",
                "qdrant": "not_implemented",
            },
        }

    return app


# Application instance for uvicorn
app = create_app()

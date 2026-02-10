"""Anki Atlas API - FastAPI application."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from packages.common.config import get_settings
from packages.common.database import check_connection, close_pool, run_migrations


class SyncRequest(BaseModel):
    """Request body for sync endpoint."""

    source: str
    run_migrations: bool = True


class SyncResponse(BaseModel):
    """Response from sync endpoint."""

    status: str
    decks_upserted: int
    models_upserted: int
    notes_upserted: int
    notes_deleted: int
    cards_upserted: int
    card_stats_upserted: int
    duration_ms: int


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    settings = get_settings()
    app.state.settings = settings
    yield
    # Shutdown
    await close_pool()


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
        """Readiness check - verifies all dependencies are available."""
        postgres_ok = await check_connection()

        # TODO: Add qdrant check
        qdrant_ok = True

        all_ok = postgres_ok and qdrant_ok

        return {
            "status": "ready" if all_ok else "not_ready",
            "checks": {
                "postgres": "ok" if postgres_ok else "failed",
                "qdrant": "ok" if qdrant_ok else "failed",
            },
        }

    @app.post("/sync", response_model=SyncResponse)
    async def sync(request: SyncRequest) -> SyncResponse:
        """Sync an Anki collection to the database.

        Args:
            request: Sync request with source path and options.

        Returns:
            Sync statistics.
        """
        from packages.anki.sync import sync_anki_collection

        source_path = Path(request.source).expanduser().resolve()

        if not source_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Collection not found: {source_path}",
            )

        # Run migrations if requested
        if request.run_migrations:
            try:
                await run_migrations()
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Migration failed: {e}",
                ) from e

        # Run sync
        try:
            stats = await sync_anki_collection(source_path)
            return SyncResponse(
                status="success",
                decks_upserted=stats.decks_upserted,
                models_upserted=stats.models_upserted,
                notes_upserted=stats.notes_upserted,
                notes_deleted=stats.notes_deleted,
                cards_upserted=stats.cards_upserted,
                card_stats_upserted=stats.card_stats_upserted,
                duration_ms=stats.duration_ms,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Sync failed: {e}",
            ) from e

    return app


# Application instance for uvicorn
app = create_app()

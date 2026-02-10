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
from packages.indexer.qdrant import close_qdrant_repository, get_qdrant_repository


class SyncRequest(BaseModel):
    """Request body for sync endpoint."""

    source: str
    run_migrations: bool = True
    index: bool = True
    force_reindex: bool = False


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
    # Indexing stats (optional)
    notes_embedded: int | None = None
    notes_skipped: int | None = None
    index_errors: list[str] | None = None


class IndexRequest(BaseModel):
    """Request body for index endpoint."""

    force_reindex: bool = False


class IndexResponse(BaseModel):
    """Response from index endpoint."""

    status: str
    notes_processed: int
    notes_embedded: int
    notes_skipped: int
    notes_deleted: int
    errors: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    settings = get_settings()
    app.state.settings = settings
    yield
    # Shutdown
    await close_pool()
    await close_qdrant_repository()


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

        # Check Qdrant connectivity
        try:
            qdrant = await get_qdrant_repository()
            qdrant_ok = await qdrant.health_check()
        except Exception:
            qdrant_ok = False

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
        """Sync an Anki collection to the database and optionally index.

        Args:
            request: Sync request with source path and options.

        Returns:
            Sync and indexing statistics.
        """
        from packages.anki.sync import sync_anki_collection
        from packages.indexer.service import index_all_notes

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

        # Run sync to PostgreSQL
        try:
            stats = await sync_anki_collection(source_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Sync failed: {e}",
            ) from e

        response = SyncResponse(
            status="success",
            decks_upserted=stats.decks_upserted,
            models_upserted=stats.models_upserted,
            notes_upserted=stats.notes_upserted,
            notes_deleted=stats.notes_deleted,
            cards_upserted=stats.cards_upserted,
            card_stats_upserted=stats.card_stats_upserted,
            duration_ms=stats.duration_ms,
        )

        # Run indexing if requested
        if request.index:
            try:
                index_stats = await index_all_notes(force_reindex=request.force_reindex)
                response.notes_embedded = index_stats.notes_embedded
                response.notes_skipped = index_stats.notes_skipped
                response.index_errors = index_stats.errors if index_stats.errors else None
            except Exception as e:
                response.index_errors = [str(e)]

        return response

    @app.post("/index", response_model=IndexResponse)
    async def index_notes(request: IndexRequest) -> IndexResponse:
        """Index notes from PostgreSQL to vector database.

        Args:
            request: Index request with options.

        Returns:
            Indexing statistics.
        """
        from packages.indexer.service import index_all_notes

        try:
            stats = await index_all_notes(force_reindex=request.force_reindex)
            return IndexResponse(
                status="success",
                notes_processed=stats.notes_processed,
                notes_embedded=stats.notes_embedded,
                notes_skipped=stats.notes_skipped,
                notes_deleted=stats.notes_deleted,
                errors=stats.errors,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Indexing failed: {e}",
            ) from e

    @app.get("/index/info", response_class=JSONResponse)
    async def index_info() -> dict[str, Any]:
        """Get information about the vector index."""
        try:
            qdrant = await get_qdrant_repository()
            info = await qdrant.get_collection_info()
            return {
                "status": "ok" if info else "not_created",
                "collection": info,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    return app


# Application instance for uvicorn
app = create_app()

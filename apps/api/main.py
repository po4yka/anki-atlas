"""Anki Atlas API - FastAPI application."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from apps.api.schemas import (
    AsyncIndexRequest,
    AsyncSyncRequest,
    DuplicateClusterItem,
    DuplicateNoteItem,
    DuplicatesResponse,
    IndexRequest,
    IndexResponse,
    JobAcceptedResponse,
    JobStatusResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SyncRequest,
    SyncResponse,
    TopicCoverageResponse,
    TopicGapItem,
    TopicGapsResponse,
)
from packages.common.config import get_settings
from packages.common.database import check_connection, close_pool, run_migrations
from packages.common.exceptions import (
    AnkiAtlasError,
    ConflictError,
    DatabaseError,
    NotFoundError,
    VectorStoreError,
)
from packages.common.logging import (
    clear_correlation_id,
    configure_logging,
    get_logger,
    set_correlation_id,
)
from packages.indexer.qdrant import close_qdrant_repository, get_qdrant_repository
from packages.jobs import (
    JobBackendUnavailableError,
    JobRecord,
    close_job_manager,
    get_job_manager,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = get_logger(module=__name__)


def _to_job_status_response(job: JobRecord) -> JobStatusResponse:
    """Map internal job record to API response."""
    return JobStatusResponse(
        job_id=job.job_id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        message=job.message,
        attempts=job.attempts,
        max_retries=job.max_retries,
        cancel_requested=job.cancel_requested,
        created_at=job.created_at,
        scheduled_for=job.scheduled_for,
        started_at=job.started_at,
        finished_at=job.finished_at,
        result=job.result,
        error=job.error,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan handler for startup/shutdown."""
    settings = get_settings()
    configure_logging(debug=settings.debug, json_output=not settings.debug)
    app.state.settings = settings
    yield
    await close_pool()
    await close_qdrant_repository()
    await close_job_manager()


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    api_key: str | None = Depends(_api_key_header),
) -> None:
    """Verify API key if ANKIATLAS_API_KEY is configured."""
    settings = get_settings()
    if settings.api_key is None:
        return
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


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

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next: Any) -> Any:
        """Add correlation ID to request context for tracing."""
        correlation_id = request.headers.get("X-Request-ID")
        set_correlation_id(correlation_id)

        try:
            response = await call_next(request)
            from packages.common.logging import get_correlation_id

            response.headers["X-Request-ID"] = get_correlation_id() or ""
            return response
        finally:
            clear_correlation_id()

    @app.exception_handler(AnkiAtlasError)
    async def ankiatlas_exception_handler(request: Request, exc: AnkiAtlasError) -> JSONResponse:
        """Handle application-specific exceptions with appropriate status codes."""
        if isinstance(exc, NotFoundError):
            status_code = 404
        elif isinstance(exc, ConflictError):
            status_code = 409
        elif isinstance(exc, (DatabaseError, VectorStoreError)):
            status_code = 503
        else:
            status_code = 500

        logger.error(
            "request_failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            status_code=status_code,
            path=str(request.url.path),
            **exc.context,
        )

        return JSONResponse(
            status_code=status_code,
            content={
                "error": type(exc).__name__,
                "message": str(exc),
                "path": str(request.url.path),
            },
        )

    @app.get("/health", response_class=JSONResponse)
    async def health() -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "0.1.0",
            "services": {
                "postgres": {"configured": True},
                "qdrant": {"configured": True},
                "redis": {"configured": True},
            },
        }

    @app.get("/ready", response_class=JSONResponse)
    async def ready() -> dict[str, Any]:
        """Readiness check - verifies all dependencies are available."""
        postgres_ok = await check_connection()

        try:
            qdrant = await get_qdrant_repository()
            qdrant_ok = await qdrant.health_check()
        except Exception as e:
            logger.warning("ready_check_qdrant_failed", error=str(e), error_type=type(e).__name__)
            qdrant_ok = False

        try:
            manager = await asyncio.wait_for(get_job_manager(), timeout=1.0)
            redis = await asyncio.wait_for(manager.connect(), timeout=1.0)
            redis_ok = bool(await asyncio.wait_for(redis.ping(), timeout=1.0))
        except Exception as e:
            logger.warning("ready_check_redis_failed", error=str(e), error_type=type(e).__name__)
            redis_ok = False

        all_ok = postgres_ok and qdrant_ok and redis_ok

        return {
            "status": "ready" if all_ok else "not_ready",
            "checks": {
                "postgres": "ok" if postgres_ok else "failed",
                "qdrant": "ok" if qdrant_ok else "failed",
                "redis": "ok" if redis_ok else "failed",
            },
        }

    @app.post("/sync", response_model=SyncResponse, dependencies=[Depends(require_api_key)])
    async def sync(request: SyncRequest) -> SyncResponse:
        """Sync an Anki collection to the database and optionally index."""
        from packages.anki.sync import sync_anki_collection
        from packages.common.exceptions import EmbeddingModelChanged
        from packages.indexer.service import index_all_notes

        source_path = Path(request.source).expanduser().resolve()

        if not source_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Collection not found: {source_path}",
            )

        if source_path.suffix not in (".anki2", ".anki21"):
            raise HTTPException(
                status_code=400,
                detail="Source must be an Anki collection file (.anki2 or .anki21)",
            )

        if request.run_migrations:
            try:
                await run_migrations()
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Migration failed: {e}",
                ) from e

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

        if request.index:
            try:
                index_stats = await index_all_notes(force_reindex=request.force_reindex)
                response.notes_embedded = index_stats.notes_embedded
                response.notes_skipped = index_stats.notes_skipped
                response.index_errors = index_stats.errors if index_stats.errors else None
            except EmbeddingModelChanged as e:
                logger.warning("sync_embedding_model_changed", error=str(e))
                response.index_errors = [str(e)]
            except Exception as e:
                logger.exception("sync_indexing_failed", source=str(source_path))
                response.index_errors = [str(e)]

        return response

    @app.post("/index", response_model=IndexResponse, dependencies=[Depends(require_api_key)])
    async def index_notes(request: IndexRequest) -> IndexResponse:
        """Index notes from PostgreSQL to vector database."""
        from packages.common.exceptions import EmbeddingModelChanged
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
        except EmbeddingModelChanged as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except Exception as e:
            logger.exception(
                "index_notes_failed",
                error_type=type(e).__name__,
                force_reindex=request.force_reindex,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Indexing failed: {e}",
            ) from e

    @app.post("/jobs/sync", response_model=JobAcceptedResponse, status_code=202, dependencies=[Depends(require_api_key)])
    async def enqueue_sync_job(request: AsyncSyncRequest) -> JobAcceptedResponse:
        """Enqueue async sync/index job and return job ID for polling."""
        source_path = Path(request.source).expanduser().resolve()
        if not source_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Collection not found: {source_path}",
            )

        run_at = request.run_at
        if run_at and run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=UTC)

        try:
            manager = await get_job_manager()
            job = await manager.enqueue_sync_job(
                payload={
                    "source": str(source_path),
                    "run_migrations": request.run_migrations,
                    "index": request.index,
                    "force_reindex": request.force_reindex,
                },
                run_at=run_at,
            )
        except JobBackendUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("enqueue_sync_job_failed", error_type=type(exc).__name__)
            raise HTTPException(
                status_code=500, detail=f"Failed to enqueue sync job: {exc}"
            ) from exc

        return JobAcceptedResponse(
            job_id=job.job_id,
            status=job.status,
            job_type=job.job_type,
            created_at=job.created_at or datetime.now(UTC),
            scheduled_for=job.scheduled_for,
            poll_url=f"/jobs/{job.job_id}",
        )

    @app.post("/jobs/index", response_model=JobAcceptedResponse, status_code=202, dependencies=[Depends(require_api_key)])
    async def enqueue_index_job(request: AsyncIndexRequest) -> JobAcceptedResponse:
        """Enqueue async indexing job and return job ID for polling."""
        run_at = request.run_at
        if run_at and run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=UTC)

        try:
            manager = await get_job_manager()
            job = await manager.enqueue_index_job(
                payload={
                    "force_reindex": request.force_reindex,
                },
                run_at=run_at,
            )
        except JobBackendUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("enqueue_index_job_failed", error_type=type(exc).__name__)
            raise HTTPException(
                status_code=500, detail=f"Failed to enqueue index job: {exc}"
            ) from exc

        return JobAcceptedResponse(
            job_id=job.job_id,
            status=job.status,
            job_type=job.job_type,
            created_at=job.created_at or datetime.now(UTC),
            scheduled_for=job.scheduled_for,
            poll_url=f"/jobs/{job.job_id}",
        )

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    async def get_job_status(job_id: str) -> JobStatusResponse:
        """Get status/progress for a background job."""
        try:
            manager = await get_job_manager()
            job = await manager.get_job(job_id)
        except JobBackendUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return _to_job_status_response(job)

    @app.post("/jobs/{job_id}/cancel", response_model=JobStatusResponse, dependencies=[Depends(require_api_key)])
    async def cancel_job(job_id: str) -> JobStatusResponse:
        """Request cancellation of a background job."""
        try:
            manager = await get_job_manager()
            job = await manager.cancel_job(job_id)
        except JobBackendUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        return _to_job_status_response(job)

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
            logger.exception("index_info_failed", error_type=type(e).__name__)
            return {
                "status": "error",
                "error": str(e),
            }

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest) -> SearchResponse:
        """Search notes using hybrid semantic + keyword search."""
        from packages.search import SearchFilters, SearchService

        filters = SearchFilters(
            deck_names=request.deck_names,
            deck_names_exclude=request.deck_names_exclude,
            tags=request.tags,
            tags_exclude=request.tags_exclude,
            model_ids=request.model_ids,
            min_ivl=request.min_ivl,
            max_lapses=request.max_lapses,
            min_reps=request.min_reps,
        )

        service = SearchService()
        result = await service.search(
            query=request.query,
            filters=filters,
            limit=request.top_k,
            semantic_weight=request.semantic_weight,
            fts_weight=request.fts_weight,
        )

        note_ids = [r.note_id for r in result.results]
        notes_details = await service.get_notes_details(note_ids)

        items: list[SearchResultItem] = []
        for r in result.results:
            detail = notes_details.get(r.note_id)
            items.append(
                SearchResultItem(
                    note_id=r.note_id,
                    rrf_score=r.rrf_score,
                    semantic_score=r.semantic_score,
                    semantic_rank=r.semantic_rank,
                    fts_score=r.fts_score,
                    fts_rank=r.fts_rank,
                    headline=r.headline,
                    rerank_score=r.rerank_score,
                    rerank_rank=r.rerank_rank,
                    sources=r.sources,
                    normalized_text=detail.normalized_text if detail else None,
                    tags=detail.tags if detail else None,
                    deck_names=detail.deck_names if detail else None,
                )
            )

        return SearchResponse(
            query=result.query,
            results=items,
            stats={
                "semantic_only": result.stats.semantic_only,
                "fts_only": result.stats.fts_only,
                "both": result.stats.both,
                "total": result.stats.total,
            },
            filters_applied=result.filters_applied,
            lexical={
                "mode": result.lexical_mode,
                "fallback_used": result.lexical_fallback_used,
                "query_suggestions": result.query_suggestions,
                "autocomplete_suggestions": result.autocomplete_suggestions,
            },
            rerank={
                "applied": result.rerank_applied,
                "model": result.rerank_model,
                "top_n": result.rerank_top_n,
            },
        )

    @app.get("/topics", response_class=JSONResponse)
    async def list_topics(
        root: str | None = None,
    ) -> dict[str, Any]:
        """Get taxonomy tree with coverage info."""
        from packages.analytics import AnalyticsService

        service = AnalyticsService()
        tree = await service.get_taxonomy_tree(root)

        return {
            "topics": tree,
            "count": len(tree),
        }

    @app.get("/topics/{topic_path:path}/coverage", response_model=TopicCoverageResponse)
    async def topic_coverage(
        topic_path: str,
        include_subtree: bool = True,
    ) -> TopicCoverageResponse:
        """Get coverage metrics for a topic."""
        from packages.analytics import AnalyticsService

        service = AnalyticsService()
        coverage = await service.get_coverage(topic_path, include_subtree)

        if not coverage:
            raise HTTPException(status_code=404, detail=f"Topic not found: {topic_path}")

        return TopicCoverageResponse(
            topic_id=coverage.topic_id,
            path=coverage.path,
            label=coverage.label,
            note_count=coverage.note_count,
            subtree_count=coverage.subtree_count,
            child_count=coverage.child_count,
            covered_children=coverage.covered_children,
            mature_count=coverage.mature_count,
            avg_confidence=coverage.avg_confidence,
            weak_notes=coverage.weak_notes,
            avg_lapses=coverage.avg_lapses,
        )

    @app.get("/topics/{topic_path:path}/gaps", response_model=TopicGapsResponse)
    async def topic_gaps(
        topic_path: str,
        min_coverage: int = 1,
    ) -> TopicGapsResponse:
        """Find gaps in topic coverage."""
        from packages.analytics import AnalyticsService

        service = AnalyticsService()
        gaps = await service.get_gaps(topic_path, min_coverage)

        gap_items = [
            TopicGapItem(
                topic_id=g.topic_id,
                path=g.path,
                label=g.label,
                description=g.description,
                gap_type=g.gap_type,
                note_count=g.note_count,
                threshold=g.threshold,
            )
            for g in gaps
        ]

        missing = sum(1 for g in gaps if g.gap_type == "missing")
        undercovered = sum(1 for g in gaps if g.gap_type == "undercovered")

        return TopicGapsResponse(
            root_path=topic_path,
            min_coverage=min_coverage,
            gaps=gap_items,
            missing_count=missing,
            undercovered_count=undercovered,
        )

    @app.get("/duplicates", response_model=DuplicatesResponse)
    async def find_duplicates(
        threshold: float = 0.92,
        max_clusters: int = 100,
        deck: str | None = None,
        tag: str | None = None,
    ) -> DuplicatesResponse:
        """Find near-duplicate notes using embedding similarity."""
        from packages.analytics import DuplicateDetector

        detector = DuplicateDetector()
        clusters, stats = await detector.find_duplicates(
            threshold=threshold,
            max_clusters=max_clusters,
            deck_filter=[deck] if deck else None,
            tag_filter=[tag] if tag else None,
        )

        cluster_items = [
            DuplicateClusterItem(
                representative_id=c.representative_id,
                representative_text=c.representative_text,
                deck_names=c.deck_names,
                tags=c.tags,
                duplicates=[
                    DuplicateNoteItem(
                        note_id=d["note_id"],
                        similarity=d["similarity"],
                        text=d["text"],
                        deck_names=d["deck_names"],
                        tags=d["tags"],
                    )
                    for d in c.duplicates
                ],
                size=c.size,
            )
            for c in clusters
        ]

        return DuplicatesResponse(
            clusters=cluster_items,
            stats={
                "notes_scanned": stats.notes_scanned,
                "clusters_found": stats.clusters_found,
                "total_duplicates": stats.total_duplicates,
                "avg_cluster_size": stats.avg_cluster_size,
            },
        )

    return app


# Application instance for uvicorn
app = create_app()

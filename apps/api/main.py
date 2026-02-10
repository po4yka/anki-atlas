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
from packages.common.logging import configure_logging, get_logger
from packages.indexer.qdrant import close_qdrant_repository, get_qdrant_repository

logger = get_logger(module=__name__)


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


class SearchRequest(BaseModel):
    """Request body for search endpoint."""

    query: str
    deck_names: list[str] | None = None
    deck_names_exclude: list[str] | None = None
    tags: list[str] | None = None
    tags_exclude: list[str] | None = None
    model_ids: list[int] | None = None
    min_ivl: int | None = None
    max_lapses: int | None = None
    min_reps: int | None = None
    top_k: int = 20
    semantic_weight: float = 1.0
    fts_weight: float = 1.0


class SearchResultItem(BaseModel):
    """A single search result."""

    note_id: int
    rrf_score: float
    semantic_score: float | None = None
    semantic_rank: int | None = None
    fts_score: float | None = None
    fts_rank: int | None = None
    headline: str | None = None
    sources: list[str]
    # Optional enriched data
    normalized_text: str | None = None
    tags: list[str] | None = None
    deck_names: list[str] | None = None


class SearchResponse(BaseModel):
    """Response from search endpoint."""

    query: str
    results: list[SearchResultItem]
    stats: dict[str, int]
    filters_applied: dict[str, Any]


class TopicItem(BaseModel):
    """A topic in the taxonomy."""

    topic_id: int
    path: str
    label: str
    description: str | None = None
    note_count: int = 0
    avg_confidence: float = 0.0
    mature_count: int = 0
    depth: int = 0


class TopicCoverageResponse(BaseModel):
    """Coverage metrics for a topic."""

    topic_id: int
    path: str
    label: str
    note_count: int
    subtree_count: int
    child_count: int
    covered_children: int
    mature_count: int
    avg_confidence: float
    weak_notes: int
    avg_lapses: float


class TopicGapItem(BaseModel):
    """A gap in topic coverage."""

    topic_id: int
    path: str
    label: str
    description: str | None
    gap_type: str
    note_count: int
    threshold: int


class TopicGapsResponse(BaseModel):
    """Response with topic gaps."""

    root_path: str
    min_coverage: int
    gaps: list[TopicGapItem]
    missing_count: int
    undercovered_count: int


class DuplicateNoteItem(BaseModel):
    """A duplicate note in a cluster."""

    note_id: int
    similarity: float
    text: str
    deck_names: list[str]
    tags: list[str]


class DuplicateClusterItem(BaseModel):
    """A cluster of duplicate notes."""

    representative_id: int
    representative_text: str
    deck_names: list[str]
    tags: list[str]
    duplicates: list[DuplicateNoteItem]
    size: int


class DuplicatesResponse(BaseModel):
    """Response from duplicates endpoint."""

    clusters: list[DuplicateClusterItem]
    stats: dict[str, Any]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    settings = get_settings()
    configure_logging(debug=settings.debug, json_output=not settings.debug)
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
            logger.warning("ready_check_qdrant_failed")
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
                logger.exception("sync_indexing_failed", source=str(source_path))
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
            logger.exception("index_info_failed")
            return {
                "status": "error",
                "error": str(e),
            }

    @app.post("/search", response_model=SearchResponse)
    async def search(request: SearchRequest) -> SearchResponse:
        """Search notes using hybrid semantic + keyword search.

        Args:
            request: Search request with query, filters, and options.

        Returns:
            Search results with score breakdown and statistics.
        """
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

        # Fetch note details for enrichment
        note_ids = [r.note_id for r in result.results]
        notes_details = await service.get_notes_details(note_ids)

        # Build response items
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
        )

    @app.get("/topics", response_class=JSONResponse)
    async def list_topics(
        root: str | None = None,
    ) -> dict[str, Any]:
        """Get taxonomy tree with coverage info.

        Args:
            root: Optional root path to filter.

        Returns:
            List of topics with coverage metrics.
        """
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
        """Get coverage metrics for a topic.

        Args:
            topic_path: Topic path (e.g., programming/python).
            include_subtree: Include child topics in metrics.

        Returns:
            Coverage metrics.
        """
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
        """Find gaps in topic coverage.

        Args:
            topic_path: Root topic path.
            min_coverage: Minimum notes for a topic to be considered covered.

        Returns:
            List of missing or undercovered topics.
        """
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
        """Find near-duplicate notes using embedding similarity.

        Args:
            threshold: Minimum similarity threshold (0-1).
            max_clusters: Maximum clusters to return.
            deck: Optional deck name filter.
            tag: Optional tag filter.

        Returns:
            Clusters of duplicate notes with statistics.
        """
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

"""API request/response schemas."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic needs runtime access
from typing import Any

from pydantic import BaseModel


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


class AsyncSyncRequest(BaseModel):
    """Request body for async sync job."""

    source: str
    run_migrations: bool = True
    index: bool = True
    force_reindex: bool = False
    run_at: datetime | None = None


class AsyncIndexRequest(BaseModel):
    """Request body for async index job."""

    force_reindex: bool = False
    run_at: datetime | None = None


class JobAcceptedResponse(BaseModel):
    """Response from async job enqueue."""

    job_id: str
    status: str
    job_type: str
    created_at: datetime
    scheduled_for: datetime | None = None
    poll_url: str


class JobStatusResponse(BaseModel):
    """Response with background job status/progress."""

    job_id: str
    job_type: str
    status: str
    progress: float
    message: str | None = None
    attempts: int
    max_retries: int
    cancel_requested: bool
    created_at: datetime | None = None
    scheduled_for: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


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
    limit: int = 20
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
    rerank_score: float | None = None
    rerank_rank: int | None = None
    sources: list[str]
    normalized_text: str | None = None
    tags: list[str] | None = None
    deck_names: list[str] | None = None


class SearchResponse(BaseModel):
    """Response from search endpoint."""

    query: str
    results: list[SearchResultItem]
    stats: dict[str, int]
    filters_applied: dict[str, Any]
    lexical: dict[str, Any] | None = None
    rerank: dict[str, Any] | None = None


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

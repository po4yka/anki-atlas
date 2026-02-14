"""Hybrid search service combining semantic and FTS."""

from dataclasses import dataclass, field
from typing import Any

from packages.common.config import Settings, get_settings
from packages.common.database import get_connection
from packages.indexer.embeddings import EmbeddingProvider, get_embedding_provider
from packages.indexer.qdrant import QdrantRepository, get_qdrant_repository
from packages.search.fts import SearchFilters, search_fts
from packages.search.fusion import FusionStats, SearchResult, reciprocal_rank_fusion


@dataclass
class HybridSearchResult:
    """Complete result from hybrid search."""

    results: list[SearchResult]
    stats: FusionStats
    query: str
    filters_applied: dict[str, Any] = field(default_factory=dict)


@dataclass
class NoteDetail:
    """Detailed note information for search results."""

    note_id: int
    model_id: int
    normalized_text: str
    tags: list[str]
    deck_names: list[str]
    mature: bool
    lapses: int
    reps: int


class SearchService:
    """Service for hybrid search across Anki notes."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        qdrant_repository: QdrantRepository | None = None,
    ) -> None:
        """Initialize search service.

        Args:
            settings: Application settings.
            embedding_provider: Embedding provider for semantic search.
            qdrant_repository: Qdrant repository for vector search.
        """
        self.settings = settings or get_settings()
        self._embedding_provider = embedding_provider
        self._qdrant_repository = qdrant_repository

    async def get_embedding_provider(self) -> EmbeddingProvider:
        """Get or create embedding provider."""
        if self._embedding_provider is None:
            self._embedding_provider = get_embedding_provider(self.settings)
        return self._embedding_provider

    async def get_qdrant_repository(self) -> QdrantRepository:
        """Get or create Qdrant repository."""
        if self._qdrant_repository is None:
            self._qdrant_repository = await get_qdrant_repository(self.settings)
        return self._qdrant_repository

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 50,
        semantic_weight: float = 1.0,
        fts_weight: float = 1.0,
        semantic_only: bool = False,
        fts_only: bool = False,
    ) -> HybridSearchResult:
        """Perform hybrid search.

        Args:
            query: Search query string.
            filters: Optional search filters.
            limit: Maximum results to return.
            semantic_weight: Weight for semantic search in RRF.
            fts_weight: Weight for FTS in RRF.
            semantic_only: Only use semantic search.
            fts_only: Only use FTS.

        Returns:
            HybridSearchResult with fused results and statistics.
        """
        if not query.strip():
            return HybridSearchResult(
                results=[],
                stats=FusionStats(),
                query=query,
            )

        # Collect results from both sources
        semantic_results: list[tuple[int, float]] = []
        fts_results: list[tuple[int, float, str | None]] = []

        # Run semantic search (unless fts_only)
        if not fts_only:
            semantic_results = await self._semantic_search(query, filters, limit * 2)

        # Run FTS search (unless semantic_only)
        if not semantic_only:
            fts_raw = await search_fts(query, filters, limit * 2, self.settings)
            fts_results = [(r.note_id, r.rank, r.headline) for r in fts_raw]

        # Fuse results
        results, stats = reciprocal_rank_fusion(
            semantic_results=semantic_results,
            fts_results=fts_results,
            limit=limit,
            semantic_weight=semantic_weight if not fts_only else 0.0,
            fts_weight=fts_weight if not semantic_only else 0.0,
        )

        # Build filters applied dict
        filters_applied: dict[str, Any] = {}
        if filters:
            if filters.deck_names:
                filters_applied["deck_names"] = filters.deck_names
            if filters.deck_names_exclude:
                filters_applied["deck_names_exclude"] = filters.deck_names_exclude
            if filters.tags:
                filters_applied["tags"] = filters.tags
            if filters.tags_exclude:
                filters_applied["tags_exclude"] = filters.tags_exclude
            if filters.model_ids:
                filters_applied["model_ids"] = filters.model_ids
            if filters.min_ivl is not None:
                filters_applied["min_ivl"] = filters.min_ivl
            if filters.max_lapses is not None:
                filters_applied["max_lapses"] = filters.max_lapses
            if filters.min_reps is not None:
                filters_applied["min_reps"] = filters.min_reps

        return HybridSearchResult(
            results=results,
            stats=stats,
            query=query,
            filters_applied=filters_applied,
        )

    async def _semantic_search(
        self,
        query: str,
        filters: SearchFilters | None,
        limit: int,
    ) -> list[tuple[int, float]]:
        """Perform semantic search using embeddings."""
        provider = await self.get_embedding_provider()
        qdrant = await self.get_qdrant_repository()

        # Embed the query
        query_vector = await provider.embed_single(query)

        # Map filters to Qdrant format
        deck_names = filters.deck_names if filters else None
        deck_names_exclude = filters.deck_names_exclude if filters else None
        tags = filters.tags if filters else None
        tags_exclude = filters.tags_exclude if filters else None
        model_ids = filters.model_ids if filters else None
        mature_only = (filters.min_ivl or 0) >= 21 if filters else False
        max_lapses = filters.max_lapses if filters else None
        min_reps = filters.min_reps if filters else None

        return await qdrant.search(
            query_vector=query_vector,
            limit=limit,
            deck_names=deck_names,
            deck_names_exclude=deck_names_exclude,
            tags=tags,
            tags_exclude=tags_exclude,
            model_ids=model_ids,
            mature_only=mature_only,
            max_lapses=max_lapses,
            min_reps=min_reps,
        )

    async def get_notes_details(
        self,
        note_ids: list[int],
    ) -> dict[int, NoteDetail]:
        """Fetch detailed note information for result enrichment.

        Args:
            note_ids: List of note IDs to fetch.

        Returns:
            Dictionary mapping note_id to NoteDetail.
        """
        if not note_ids:
            return {}

        async with get_connection(self.settings) as conn:
            result = await conn.execute(
                """
                SELECT
                    n.note_id,
                    n.model_id,
                    n.normalized_text,
                    n.tags,
                    COALESCE(
                        array_agg(DISTINCT d.name) FILTER (WHERE d.name IS NOT NULL),
                        '{}'::text[]
                    ) as deck_names,
                    COALESCE(MAX(c.ivl) >= 21, false) as mature,
                    COALESCE(SUM(c.lapses), 0) as lapses,
                    COALESCE(SUM(c.reps), 0) as reps
                FROM notes n
                LEFT JOIN cards c ON c.note_id = n.note_id
                LEFT JOIN decks d ON d.deck_id = c.deck_id
                WHERE n.note_id = ANY(%(note_ids)s)
                GROUP BY n.note_id, n.model_id, n.normalized_text, n.tags
                """,
                {"note_ids": note_ids},
            )

            notes: dict[int, NoteDetail] = {}
            async for row in result:
                notes[row["note_id"]] = NoteDetail(
                    note_id=row["note_id"],
                    model_id=row["model_id"],
                    normalized_text=row["normalized_text"],
                    tags=row["tags"] or [],
                    deck_names=row["deck_names"] or [],
                    mature=row["mature"],
                    lapses=row["lapses"],
                    reps=row["reps"],
                )

            return notes


async def hybrid_search(
    query: str,
    filters: SearchFilters | None = None,
    limit: int = 50,
    settings: Settings | None = None,
) -> HybridSearchResult:
    """Convenience function for hybrid search.

    Args:
        query: Search query.
        filters: Optional filters.
        limit: Maximum results.
        settings: Application settings.

    Returns:
        HybridSearchResult with fused results.
    """
    service = SearchService(settings)
    return await service.search(query, filters, limit)

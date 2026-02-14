"""Hybrid search service combining semantic and FTS."""

from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from packages.common.config import Settings, get_settings
from packages.common.database import get_connection
from packages.common.logging import get_logger
from packages.indexer.embeddings import EmbeddingProvider, get_embedding_provider
from packages.indexer.qdrant import QdrantRepository, get_qdrant_repository
from packages.search.fts import SearchFilters, search_lexical
from packages.search.fusion import FusionStats, SearchResult, reciprocal_rank_fusion
from packages.search.reranker import CrossEncoderReranker, Reranker

logger = get_logger(module=__name__)


@dataclass
class HybridSearchResult:
    """Complete result from hybrid search."""

    results: list[SearchResult]
    stats: FusionStats
    query: str
    filters_applied: dict[str, Any] = field(default_factory=dict)
    lexical_mode: str = "none"
    lexical_fallback_used: bool = False
    query_suggestions: list[str] = field(default_factory=list)
    autocomplete_suggestions: list[str] = field(default_factory=list)
    rerank_applied: bool = False
    rerank_model: str | None = None
    rerank_top_n: int | None = None


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
        reranker: Reranker | None = None,
    ) -> None:
        """Initialize search service.

        Args:
            settings: Application settings.
            embedding_provider: Embedding provider for semantic search.
            qdrant_repository: Qdrant repository for vector search.
            reranker: Optional reranker implementation.
        """
        self.settings = settings or get_settings()
        self._embedding_provider = embedding_provider
        self._qdrant_repository = qdrant_repository
        self._reranker = reranker
        self._reranker_unavailable_logged = False

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

    async def get_reranker(self) -> Reranker:
        """Get or create reranker."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model_name=self.settings.rerank_model,
                batch_size=self.settings.rerank_batch_size,
            )
        return self._reranker

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        limit: int = 50,
        semantic_weight: float = 1.0,
        fts_weight: float = 1.0,
        semantic_only: bool = False,
        fts_only: bool = False,
        rerank: bool | None = None,
        rerank_top_n: int | None = None,
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
            rerank: Override config and enable/disable second-stage reranking.
            rerank_top_n: Override configured rerank candidate count.

        Returns:
            HybridSearchResult with fused results and statistics.
        """
        if not query.strip():
            return HybridSearchResult(
                results=[],
                stats=FusionStats(),
                query=query,
            )

        rerank_enabled = self.settings.rerank_enabled if rerank is None else rerank
        rerank_candidate_limit = rerank_top_n or self.settings.rerank_top_n
        candidate_limit = max(limit, rerank_candidate_limit) if rerank_enabled else limit
        retrieval_limit = candidate_limit * 2

        # Collect results from both sources
        semantic_results: list[tuple[int, float]] = []
        fts_results: list[tuple[int, float, str | None]] = []

        # Run semantic search (unless fts_only)
        if not fts_only:
            semantic_results = await self._semantic_search(query, filters, retrieval_limit)

        lexical_mode = "none"
        lexical_fallback_used = False
        query_suggestions: list[str] = []
        autocomplete_suggestions: list[str] = []

        # Run lexical search (unless semantic_only)
        if not semantic_only:
            lexical = await search_lexical(query, filters, retrieval_limit, self.settings)
            lexical_mode = lexical.mode
            lexical_fallback_used = lexical.used_fallback
            query_suggestions = lexical.query_suggestions
            autocomplete_suggestions = lexical.autocomplete_suggestions
            fts_results = [(r.note_id, r.rank, r.headline) for r in lexical.results]

        # Fuse results
        results, stats = reciprocal_rank_fusion(
            semantic_results=semantic_results,
            fts_results=fts_results,
            limit=candidate_limit,
            semantic_weight=semantic_weight if not fts_only else 0.0,
            fts_weight=fts_weight if not semantic_only else 0.0,
        )

        rerank_applied = False
        if rerank_enabled:
            results, rerank_applied = await self._rerank_results(
                query=query,
                results=results,
                limit=limit,
                rerank_top_n=rerank_candidate_limit,
            )
        else:
            results = results[:limit]

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
            lexical_mode=lexical_mode,
            lexical_fallback_used=lexical_fallback_used,
            query_suggestions=query_suggestions,
            autocomplete_suggestions=autocomplete_suggestions,
            rerank_applied=rerank_applied,
            rerank_model=self.settings.rerank_model if rerank_enabled else None,
            rerank_top_n=rerank_candidate_limit if rerank_enabled else None,
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
        query_sparse_vector = QdrantRepository.text_to_sparse_vector(query)
        prefetch_limit = max(limit * 3, 30)

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
            query_sparse_vector=query_sparse_vector,
            limit=limit,
            prefetch_limit=prefetch_limit,
            deck_names=deck_names,
            deck_names_exclude=deck_names_exclude,
            tags=tags,
            tags_exclude=tags_exclude,
            model_ids=model_ids,
            mature_only=mature_only,
            max_lapses=max_lapses,
            min_reps=min_reps,
        )

    async def _rerank_results(
        self,
        query: str,
        results: list[SearchResult],
        limit: int,
        rerank_top_n: int,
    ) -> tuple[list[SearchResult], bool]:
        """Rerank top-N hybrid candidates with a cross-encoder."""
        if len(results) <= 1:
            return results[:limit], False

        top_n = min(max(limit, rerank_top_n), len(results))
        candidates = results[:top_n]
        note_ids = [r.note_id for r in candidates]
        note_details = await self.get_notes_details(note_ids)

        docs_for_scoring: list[tuple[int, str]] = []
        for candidate in candidates:
            detail = note_details.get(candidate.note_id)
            if detail and detail.normalized_text.strip():
                docs_for_scoring.append((candidate.note_id, detail.normalized_text))

        if len(docs_for_scoring) <= 1:
            return results[:limit], False

        try:
            reranker = await self.get_reranker()
            rerank_scores = await reranker.rerank(query, docs_for_scoring)
        except Exception as e:
            if not self._reranker_unavailable_logged:
                with suppress(Exception):
                    logger.warning(
                        "rerank_unavailable",
                        model=self.settings.rerank_model,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                self._reranker_unavailable_logged = True
            return results[:limit], False

        score_by_note = dict(rerank_scores)
        scored_candidates: list[SearchResult] = []
        unscored_candidates: list[SearchResult] = []
        for candidate in candidates:
            candidate.rerank_score = score_by_note.get(candidate.note_id)
            if candidate.rerank_score is None:
                unscored_candidates.append(candidate)
            else:
                scored_candidates.append(candidate)

        if not scored_candidates:
            return results[:limit], False

        scored_candidates.sort(
            key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"),
            reverse=True,
        )
        for rank, candidate in enumerate(scored_candidates, start=1):
            candidate.rerank_rank = rank

        reordered = scored_candidates + unscored_candidates + results[top_n:]
        return reordered[:limit], True

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

"""Reciprocal Rank Fusion for hybrid search."""

from dataclasses import dataclass


@dataclass
class SearchResult:
    """A search result with score breakdown."""

    note_id: int
    rrf_score: float
    semantic_score: float | None = None
    semantic_rank: int | None = None
    fts_score: float | None = None
    fts_rank: int | None = None
    headline: str | None = None

    @property
    def sources(self) -> list[str]:
        """Return list of sources that contributed to this result."""
        sources = []
        if self.semantic_score is not None:
            sources.append("semantic")
        if self.fts_score is not None:
            sources.append("fts")
        return sources


@dataclass
class FusionStats:
    """Statistics about the fusion operation."""

    semantic_only: int = 0
    fts_only: int = 0
    both: int = 0
    total: int = 0


def reciprocal_rank_fusion(
    semantic_results: list[tuple[int, float]],
    fts_results: list[tuple[int, float, str | None]],
    k: int = 60,
    limit: int = 50,
    semantic_weight: float = 1.0,
    fts_weight: float = 1.0,
) -> tuple[list[SearchResult], FusionStats]:
    """Fuse semantic and FTS results using Reciprocal Rank Fusion.

    RRF score = Σ weight / (k + rank)

    Args:
        semantic_results: List of (note_id, score) from semantic search.
        fts_results: List of (note_id, score, headline) from FTS.
        k: RRF constant (default 60, higher = more weight to lower ranks).
        limit: Maximum results to return.
        semantic_weight: Weight for semantic search contribution.
        fts_weight: Weight for FTS contribution.

    Returns:
        Tuple of (fused results, fusion statistics).
    """
    # Build maps for quick lookup
    semantic_map: dict[int, tuple[int, float]] = {}  # note_id -> (rank, score)
    for rank, (note_id, score) in enumerate(semantic_results, start=1):
        semantic_map[note_id] = (rank, score)

    fts_map: dict[int, tuple[int, float, str | None]] = {}  # note_id -> (rank, score, headline)
    for rank, (note_id, score, hl) in enumerate(fts_results, start=1):
        fts_map[note_id] = (rank, score, hl)

    # Collect all unique note IDs
    all_note_ids = set(semantic_map.keys()) | set(fts_map.keys())

    # Calculate RRF scores
    results: list[SearchResult] = []
    stats = FusionStats(total=len(all_note_ids))

    for note_id in all_note_ids:
        rrf_score = 0.0
        semantic_score: float | None = None
        semantic_rank: int | None = None
        fts_score: float | None = None
        fts_rank: int | None = None
        headline: str | None = None

        if note_id in semantic_map:
            rank, score = semantic_map[note_id]
            semantic_rank = rank
            semantic_score = score
            rrf_score += semantic_weight / (k + rank)

        if note_id in fts_map:
            rank, score, hl = fts_map[note_id]
            fts_rank = rank
            fts_score = score
            headline = hl
            rrf_score += fts_weight / (k + rank)

        # Track statistics
        if semantic_score is not None and fts_score is not None:
            stats.both += 1
        elif semantic_score is not None:
            stats.semantic_only += 1
        else:
            stats.fts_only += 1

        results.append(
            SearchResult(
                note_id=note_id,
                rrf_score=rrf_score,
                semantic_score=semantic_score,
                semantic_rank=semantic_rank,
                fts_score=fts_score,
                fts_rank=fts_rank,
                headline=headline,
            )
        )

    # Sort by RRF score descending
    results.sort(key=lambda r: r.rrf_score, reverse=True)

    return results[:limit], stats

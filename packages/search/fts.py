"""Full-text search using PostgreSQL."""

from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection

from packages.common.config import Settings, get_settings
from packages.common.database import get_connection


@dataclass
class FTSResult:
    """Result from full-text search."""

    note_id: int
    rank: float  # ts_rank score
    headline: str | None = None  # Optional highlighted snippet


@dataclass
class SearchFilters:
    """Filters for search queries."""

    deck_names: list[str] | None = None
    deck_names_exclude: list[str] | None = None
    tags: list[str] | None = None
    tags_exclude: list[str] | None = None
    model_ids: list[int] | None = None
    min_ivl: int | None = None  # Maturity threshold
    max_lapses: int | None = None
    min_reps: int | None = None


async def search_fts(
    query: str,
    filters: SearchFilters | None = None,
    limit: int = 50,
    settings: Settings | None = None,
) -> list[FTSResult]:
    """Search notes using PostgreSQL full-text search.

    Args:
        query: Search query string.
        filters: Optional filters to apply.
        limit: Maximum number of results.
        settings: Application settings.

    Returns:
        List of FTSResult ordered by relevance.
    """
    if not query.strip():
        return []

    settings = settings or get_settings()

    async with get_connection(settings) as conn:
        return await _execute_fts_query(conn, query, filters, limit)


async def _execute_fts_query(
    conn: AsyncConnection[dict[str, Any]],
    query: str,
    filters: SearchFilters | None,
    limit: int,
) -> list[FTSResult]:
    """Execute the FTS query with filters."""
    # Build the base query
    sql_parts = [
        """
        SELECT
            n.note_id,
            ts_rank(
                to_tsvector('english', n.normalized_text),
                websearch_to_tsquery('english', %(query)s)
            ) as rank,
            ts_headline(
                'english',
                n.normalized_text,
                websearch_to_tsquery('english', %(query)s),
                'MaxWords=30, MinWords=15, StartSel=<<, StopSel=>>'
            ) as headline
        FROM notes n
        """
    ]

    params: dict[str, Any] = {"query": query, "limit": limit}

    # Add joins for card-based filters
    if filters and _needs_card_join(filters):
        sql_parts.append(
            """
            LEFT JOIN cards c ON c.note_id = n.note_id
            LEFT JOIN decks d ON d.deck_id = c.deck_id
            """
        )

    # WHERE clause
    where_clauses = [
        "n.deleted_at IS NULL",
        "to_tsvector('english', n.normalized_text) @@ websearch_to_tsquery('english', %(query)s)",
    ]

    if filters:
        _add_filter_clauses(filters, where_clauses, params)

    sql_parts.append("WHERE " + " AND ".join(where_clauses))

    # GROUP BY for card joins
    if filters and _needs_card_join(filters):
        sql_parts.append("GROUP BY n.note_id, n.normalized_text")

    # ORDER and LIMIT
    sql_parts.append("ORDER BY rank DESC LIMIT %(limit)s")

    sql = "\n".join(sql_parts)

    result = await conn.execute(sql, params)
    rows = await result.fetchall()

    return [
        FTSResult(
            note_id=row["note_id"],
            rank=float(row["rank"]),
            headline=row["headline"],
        )
        for row in rows
    ]


def _needs_card_join(filters: SearchFilters) -> bool:
    """Check if filters require joining cards table."""
    return any(
        [
            filters.deck_names,
            filters.deck_names_exclude,
            filters.min_ivl is not None,
            filters.max_lapses is not None,
            filters.min_reps is not None,
        ]
    )


def _add_filter_clauses(
    filters: SearchFilters,
    where_clauses: list[str],
    params: dict[str, Any],
) -> None:
    """Add filter conditions to WHERE clause."""
    # Tag filters
    if filters.tags:
        where_clauses.append("n.tags && %(tags)s")
        params["tags"] = filters.tags

    if filters.tags_exclude:
        where_clauses.append("NOT (n.tags && %(tags_exclude)s)")
        params["tags_exclude"] = filters.tags_exclude

    # Model filter
    if filters.model_ids:
        where_clauses.append("n.model_id = ANY(%(model_ids)s)")
        params["model_ids"] = filters.model_ids

    # Deck filters (require card join)
    if filters.deck_names:
        where_clauses.append("d.name = ANY(%(deck_names)s)")
        params["deck_names"] = filters.deck_names

    if filters.deck_names_exclude:
        where_clauses.append("d.name IS NULL OR NOT (d.name = ANY(%(deck_names_exclude)s))")
        params["deck_names_exclude"] = filters.deck_names_exclude

    # Card stat filters
    if filters.min_ivl is not None:
        where_clauses.append("c.ivl >= %(min_ivl)s")
        params["min_ivl"] = filters.min_ivl

    if filters.max_lapses is not None:
        where_clauses.append("c.lapses <= %(max_lapses)s")
        params["max_lapses"] = filters.max_lapses

    if filters.min_reps is not None:
        where_clauses.append("c.reps >= %(min_reps)s")
        params["min_reps"] = filters.min_reps

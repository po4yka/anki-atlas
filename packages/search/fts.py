"""Lexical search using PostgreSQL FTS + trigram fallbacks."""

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from psycopg import AsyncConnection

from packages.common.config import Settings, get_settings
from packages.common.database import get_connection


@dataclass
class FTSResult:
    """Result from lexical search."""

    note_id: int
    rank: float
    headline: str | None = None  # Optional highlighted snippet
    source: Literal["fts", "fuzzy", "autocomplete"] = "fts"


@dataclass
class LexicalSearchResult:
    """Result bundle for lexical search with fallback metadata."""

    results: list[FTSResult]
    mode: Literal["fts", "fuzzy", "autocomplete", "none"] = "none"
    used_fallback: bool = False
    query_suggestions: list[str] = field(default_factory=list)
    autocomplete_suggestions: list[str] = field(default_factory=list)


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
    """Search notes lexically and return results only.

    This compatibility wrapper keeps legacy callers unchanged.
    """
    lexical = await search_lexical(query, filters, limit, settings)
    return lexical.results


async def search_lexical(
    query: str,
    filters: SearchFilters | None = None,
    limit: int = 50,
    settings: Settings | None = None,
) -> LexicalSearchResult:
    """Search notes with FTS, fuzzy fallback, and autocomplete fallback.

    Args:
        query: Search query string.
        filters: Optional filters to apply.
        limit: Maximum number of results.
        settings: Application settings.

    Returns:
        LexicalSearchResult with results and fallback diagnostics.
    """
    if not query.strip():
        return LexicalSearchResult(results=[], mode="none")

    settings = settings or get_settings()

    async with get_connection(settings) as conn:
        fts_results = await _execute_fts_query(conn, query, filters, limit)
        if fts_results:
            return LexicalSearchResult(results=fts_results, mode="fts")

        fuzzy_results = await _execute_fuzzy_query(conn, query, filters, limit)
        query_suggestions = await _fetch_query_suggestions(conn, query, filters, limit=5)
        autocomplete_suggestions = await _fetch_autocomplete_suggestions(
            conn, query, filters, limit=5
        )

        if fuzzy_results:
            return LexicalSearchResult(
                results=fuzzy_results,
                mode="fuzzy",
                used_fallback=True,
                query_suggestions=query_suggestions,
                autocomplete_suggestions=autocomplete_suggestions,
            )

        autocomplete_results = await _execute_autocomplete_query(conn, query, filters, limit)
        mode: Literal["autocomplete", "none"] = "autocomplete" if autocomplete_results else "none"
        return LexicalSearchResult(
            results=autocomplete_results,
            mode=mode,
            used_fallback=bool(autocomplete_results),
            query_suggestions=query_suggestions,
            autocomplete_suggestions=autocomplete_suggestions,
        )


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
            source="fts",
        )
        for row in rows
    ]


async def _execute_fuzzy_query(
    conn: AsyncConnection[dict[str, Any]],
    query: str,
    filters: SearchFilters | None,
    limit: int,
) -> list[FTSResult]:
    """Execute trigram-based fuzzy lexical search."""
    sql_parts = [
        """
        SELECT
            n.note_id,
            GREATEST(
                similarity(n.normalized_text, %(query)s),
                word_similarity(%(query)s, n.normalized_text)
            ) as rank,
            NULL::text as headline
        FROM notes n
        """
    ]

    params: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "similarity_threshold": 0.15,
        "word_similarity_threshold": 0.2,
    }

    joined = filters is not None and _needs_card_join(filters)
    if joined:
        sql_parts.append(
            """
            LEFT JOIN cards c ON c.note_id = n.note_id
            LEFT JOIN decks d ON d.deck_id = c.deck_id
            """
        )

    where_clauses = [
        "n.deleted_at IS NULL",
        """(
            n.normalized_text %% %(query)s
            OR similarity(n.normalized_text, %(query)s) >= %(similarity_threshold)s
            OR word_similarity(%(query)s, n.normalized_text) >= %(word_similarity_threshold)s
        )""",
    ]

    if filters:
        _add_filter_clauses(filters, where_clauses, params)

    sql_parts.append("WHERE " + " AND ".join(where_clauses))
    if joined:
        sql_parts.append("GROUP BY n.note_id, n.normalized_text")
    sql_parts.append("ORDER BY rank DESC LIMIT %(limit)s")

    result = await conn.execute("\n".join(sql_parts), params)
    rows = await result.fetchall()

    return [
        FTSResult(
            note_id=row["note_id"],
            rank=float(row["rank"]),
            headline=row["headline"],
            source="fuzzy",
        )
        for row in rows
    ]


async def _execute_autocomplete_query(
    conn: AsyncConnection[dict[str, Any]],
    query: str,
    filters: SearchFilters | None,
    limit: int,
) -> list[FTSResult]:
    """Execute autocomplete fallback query when FTS and fuzzy both miss."""
    prefix = _extract_query_prefix(query)
    if len(prefix) < 2:
        return []

    sql_parts = [
        """
        SELECT
            n.note_id,
            similarity(n.normalized_text, %(prefix)s) as rank,
            NULL::text as headline
        FROM notes n
        """
    ]

    params: dict[str, Any] = {
        "prefix": prefix,
        "prefix_like": f"{prefix}%",
        "limit": limit,
    }

    joined = filters is not None and _needs_card_join(filters)
    if joined:
        sql_parts.append(
            """
            LEFT JOIN cards c ON c.note_id = n.note_id
            LEFT JOIN decks d ON d.deck_id = c.deck_id
            """
        )

    where_clauses = [
        "n.deleted_at IS NULL",
        """
        EXISTS (
            SELECT 1
            FROM regexp_split_to_table(n.normalized_text, E'\\s+') AS token
            WHERE lower(token) LIKE %(prefix_like)s
        )
        """,
    ]
    if filters:
        _add_filter_clauses(filters, where_clauses, params)

    sql_parts.append("WHERE " + " AND ".join(where_clauses))
    if joined:
        sql_parts.append("GROUP BY n.note_id, n.normalized_text")
    sql_parts.append("ORDER BY rank DESC LIMIT %(limit)s")

    result = await conn.execute("\n".join(sql_parts), params)
    rows = await result.fetchall()

    return [
        FTSResult(
            note_id=row["note_id"],
            rank=float(row["rank"]),
            headline=row["headline"],
            source="autocomplete",
        )
        for row in rows
    ]


async def _fetch_query_suggestions(
    conn: AsyncConnection[dict[str, Any]],
    query: str,
    filters: SearchFilters | None,
    limit: int,
) -> list[str]:
    """Suggest nearest known phrases for typo-prone queries."""
    sql_parts = [
        """
        SELECT
            LEFT(n.normalized_text, 80) as suggestion,
            n.normalized_text <-> %(query)s as distance
        FROM notes n
        """
    ]

    params: dict[str, Any] = {"query": query, "limit": limit}
    joined = filters is not None and _needs_card_join(filters)
    if joined:
        sql_parts.append(
            """
            LEFT JOIN cards c ON c.note_id = n.note_id
            LEFT JOIN decks d ON d.deck_id = c.deck_id
            """
        )

    where_clauses = [
        "n.deleted_at IS NULL",
        "n.normalized_text <> ''",
    ]
    if filters:
        _add_filter_clauses(filters, where_clauses, params)

    sql_parts.append("WHERE " + " AND ".join(where_clauses))
    if joined:
        sql_parts.append("GROUP BY n.note_id, n.normalized_text")
    sql_parts.append("ORDER BY distance ASC LIMIT %(limit)s")

    result = await conn.execute("\n".join(sql_parts), params)
    rows = await result.fetchall()
    return [str(row["suggestion"]) for row in rows if row.get("suggestion")]


async def _fetch_autocomplete_suggestions(
    conn: AsyncConnection[dict[str, Any]],
    query: str,
    filters: SearchFilters | None,
    limit: int,
) -> list[str]:
    """Return likely word completions for the query prefix."""
    prefix = _extract_query_prefix(query)
    if len(prefix) < 2:
        return []

    sql_parts = [
        """
        SELECT suggestion
        FROM (
            SELECT
                lower(token) as suggestion,
                MAX(similarity(lower(token), %(prefix)s)) as score
            FROM notes n
            CROSS JOIN LATERAL regexp_split_to_table(n.normalized_text, E'\\s+') AS token
        """
    ]

    params: dict[str, Any] = {
        "prefix": prefix,
        "prefix_like": f"{prefix}%",
        "limit": limit,
    }

    joined = filters is not None and _needs_card_join(filters)
    if joined:
        sql_parts.append(
            """
            LEFT JOIN cards c ON c.note_id = n.note_id
            LEFT JOIN decks d ON d.deck_id = c.deck_id
            """
        )

    where_clauses = [
        "n.deleted_at IS NULL",
        "length(token) >= 2",
        "lower(token) LIKE %(prefix_like)s",
    ]
    if filters:
        _add_filter_clauses(filters, where_clauses, params)

    sql_parts.append("WHERE " + " AND ".join(where_clauses))
    sql_parts.append("GROUP BY lower(token)")
    sql_parts.append(
        """
        ) suggestions
        ORDER BY score DESC, suggestion ASC
        LIMIT %(limit)s
        """
    )

    result = await conn.execute("\n".join(sql_parts), params)
    rows = await result.fetchall()
    return [str(row["suggestion"]) for row in rows if row.get("suggestion")]


def _extract_query_prefix(query: str) -> str:
    """Extract trailing token for autocomplete purposes."""
    tokens = re.findall(r"[a-zA-Z0-9_]+", query.lower())
    return tokens[-1] if tokens else ""


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

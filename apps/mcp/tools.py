"""MCP tools wrapping Anki Atlas services."""

from __future__ import annotations

import asyncio
from typing import Annotated

from pydantic import Field

from apps.mcp.server import mcp
from packages.common.exceptions import (
    DatabaseConnectionError,
    VectorStoreConnectionError,
)
from packages.common.logging import get_logger

logger = get_logger(module=__name__)

# Default timeout for MCP tool operations (30 seconds)
TOOL_TIMEOUT_SECONDS = 30


def _format_error(error: Exception, operation: str) -> str:
    """Format an error message with actionable guidance.

    Args:
        error: The exception that occurred.
        operation: Description of the operation that failed.

    Returns:
        User-friendly error message with guidance.
    """
    error_type = type(error).__name__

    if isinstance(error, DatabaseConnectionError):
        return (
            f"**Error**: Database unavailable during {operation}.\n\n"
            "**Suggested actions:**\n"
            "1. Ensure PostgreSQL is running\n"
            "2. Run `anki-atlas migrate` to initialize the database\n"
            "3. Check ANKIATLAS_POSTGRES_URL in your environment"
        )
    elif isinstance(error, VectorStoreConnectionError):
        return (
            f"**Error**: Vector database unavailable during {operation}.\n\n"
            "**Suggested actions:**\n"
            "1. Ensure Qdrant is running\n"
            "2. Check ANKIATLAS_QDRANT_URL in your environment\n"
            "3. Run `anki-atlas sync --source <collection>` to index notes"
        )
    elif isinstance(error, TimeoutError):
        return (
            f"**Error**: {operation} timed out after {TOOL_TIMEOUT_SECONDS}s.\n\n"
            "**Suggested actions:**\n"
            "1. Try a more specific query\n"
            "2. Reduce the result limit\n"
            "3. Check if services are under heavy load"
        )
    else:
        return f"**Error**: {operation} failed - {error_type}: {error}"


@mcp.tool()
async def ankiatlas_search(
    query: Annotated[str, Field(description="Search query text")],
    limit: Annotated[int, Field(description="Maximum results to return", ge=1, le=100)] = 20,
    deck_filter: Annotated[
        list[str] | None, Field(description="Filter by deck names (optional)")
    ] = None,
    tag_filter: Annotated[list[str] | None, Field(description="Filter by tags (optional)")] = None,
    semantic_only: Annotated[bool, Field(description="Use only semantic search (no FTS)")] = False,
    fts_only: Annotated[bool, Field(description="Use only full-text search (no semantic)")] = False,
) -> str:
    """Search Anki notes using hybrid semantic and full-text search.

    Combines embedding-based semantic search with PostgreSQL full-text search
    using Reciprocal Rank Fusion (RRF) for optimal results.

    Returns matching notes with relevance scores, previews, tags, and deck info.

    Constraints:
        - limit: 1-100 results
        - query: non-empty string
    """
    # Lazy imports to avoid import-time side effects
    from apps.mcp.formatters import format_search_result
    from packages.search.fts import SearchFilters
    from packages.search.service import SearchService

    try:
        async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
            service = SearchService()

            # Build filters
            filters = None
            if deck_filter or tag_filter:
                filters = SearchFilters(
                    deck_names=deck_filter,
                    tags=tag_filter,
                )

            # Perform search
            result = await service.search(
                query=query,
                filters=filters,
                limit=limit,
                semantic_only=semantic_only,
                fts_only=fts_only,
            )

            # Enrich with note details
            note_ids = [r.note_id for r in result.results]
            note_details = await service.get_notes_details(note_ids)

            return format_search_result(result, note_details)

    except Exception as e:
        logger.exception("search_failed", query=query, limit=limit, error_type=type(e).__name__)
        return _format_error(e, "search")


@mcp.tool()
async def ankiatlas_topic_coverage(
    topic_path: Annotated[str, Field(description="Topic path to analyze (e.g., 'math/calculus')")],
    include_subtree: Annotated[bool, Field(description="Include child topics in metrics")] = True,
) -> str:
    """Get coverage metrics for a topic in your Anki collection.

    Shows how well a topic is covered: note counts, maturity metrics,
    child topic coverage, and quality indicators (lapses, weak notes).

    Use this to understand learning progress in a specific area.

    Constraints:
        - topic_path: valid topic path from your taxonomy (use '/' separators)
    """
    from apps.mcp.formatters import format_coverage_result
    from packages.analytics.service import AnalyticsService

    try:
        async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
            service = AnalyticsService()
            coverage = await service.get_coverage(topic_path, include_subtree)

            if coverage is None:
                return f"**Error**: Topic '{topic_path}' not found in taxonomy."

            return format_coverage_result(coverage)

    except Exception as e:
        logger.exception(
            "coverage_analysis_failed", topic_path=topic_path, error_type=type(e).__name__
        )
        return _format_error(e, "coverage analysis")


@mcp.tool()
async def ankiatlas_topic_gaps(
    topic_path: Annotated[str, Field(description="Root topic path to analyze (e.g., 'math')")],
    min_coverage: Annotated[
        int, Field(description="Minimum notes for a topic to be considered covered", ge=1)
    ] = 1,
) -> str:
    """Find knowledge gaps in topic coverage.

    Identifies topics that are missing (zero notes) or undercovered
    (fewer notes than the threshold).

    Use this to discover areas that need more flashcards.

    Constraints:
        - topic_path: valid root topic path from your taxonomy
        - min_coverage: minimum 1 note
    """
    from apps.mcp.formatters import format_gaps_result
    from packages.analytics.service import AnalyticsService

    try:
        async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
            service = AnalyticsService()
            gaps = await service.get_gaps(topic_path, min_coverage)

            return format_gaps_result(gaps, topic_path)

    except Exception as e:
        logger.exception(
            "gap_analysis_failed",
            topic_path=topic_path,
            min_coverage=min_coverage,
            error_type=type(e).__name__,
        )
        return _format_error(e, "gap analysis")


@mcp.tool()
async def ankiatlas_duplicates(
    threshold: Annotated[
        float, Field(description="Similarity threshold (0-1, higher = stricter)", ge=0.5, le=1.0)
    ] = 0.92,
    max_clusters: Annotated[
        int, Field(description="Maximum duplicate clusters to return", ge=1, le=500)
    ] = 50,
    deck_filter: Annotated[
        list[str] | None, Field(description="Filter by deck names (optional)")
    ] = None,
    tag_filter: Annotated[list[str] | None, Field(description="Filter by tags (optional)")] = None,
) -> str:
    """Find near-duplicate notes in your Anki collection.

    Uses embedding similarity to detect notes with very similar content.
    Returns clusters of duplicates with similarity scores and previews.

    Use this to clean up redundant flashcards.

    Constraints:
        - threshold: 0.5-1.0 (higher = stricter matching)
        - max_clusters: 1-500 clusters
    """
    from apps.mcp.formatters import format_duplicates_result
    from packages.analytics.duplicates import DuplicateDetector

    try:
        async with asyncio.timeout(TOOL_TIMEOUT_SECONDS):
            detector = DuplicateDetector()
            clusters, stats = await detector.find_duplicates(
                threshold=threshold,
                max_clusters=max_clusters,
                deck_filter=deck_filter,
                tag_filter=tag_filter,
            )

            return format_duplicates_result(clusters, stats)

    except Exception as e:
        logger.exception(
            "duplicate_detection_failed",
            threshold=threshold,
            max_clusters=max_clusters,
            error_type=type(e).__name__,
        )
        return _format_error(e, "duplicate detection")


@mcp.tool()
async def ankiatlas_sync(
    collection_path: Annotated[str, Field(description="Path to Anki collection.anki2 file")],
    run_index: Annotated[
        bool, Field(description="Also rebuild the vector index after sync")
    ] = False,
) -> str:
    """Sync an Anki collection to the search index.

    Reads the Anki SQLite database and syncs all decks, models, notes,
    and cards to PostgreSQL for searching and analysis.

    Optionally rebuilds the vector index for semantic search.

    Constraints:
        - collection_path: must point to a valid .anki2 file
    """
    from pathlib import Path

    from apps.mcp.formatters import format_sync_result
    from packages.anki.sync import sync_anki_collection

    try:
        # Validate path exists
        path = Path(collection_path)
        if not path.exists():
            return (
                f"**Error**: Collection file not found: {collection_path}\n\n"
                "**Suggested actions:**\n"
                "1. Check the file path is correct\n"
                "2. Ensure Anki is not running (it may lock the database)\n"
                "3. Try using the full absolute path"
            )

        if not path.name.endswith(".anki2"):
            return (
                f"**Error**: Expected .anki2 file, got: {path.name}\n\n"
                "The collection file is typically located at:\n"
                "- macOS: ~/Library/Application Support/Anki2/<profile>/collection.anki2\n"
                "- Linux: ~/.local/share/Anki2/<profile>/collection.anki2\n"
                "- Windows: %APPDATA%\\Anki2\\<profile>\\collection.anki2"
            )

        # Run sync with timeout (sync can be slow for large collections)
        async with asyncio.timeout(120):  # 2 minute timeout for sync
            stats = await sync_anki_collection(path)
            result = format_sync_result(stats)

        # Optionally run indexing
        if run_index:
            from packages.indexer.service import index_all_notes

            result += "\n\n---\n\n"
            result += "## Indexing\n\n"

            try:
                async with asyncio.timeout(300):  # 5 minute timeout for indexing
                    index_stats = await index_all_notes()
                    result += f"- **Notes processed**: {index_stats.notes_processed}\n"
                    result += f"- **Notes embedded**: {index_stats.notes_embedded}\n"
                    result += f"- **Notes skipped**: {index_stats.notes_skipped}\n"
                    if index_stats.errors:
                        result += f"- **Errors**: {len(index_stats.errors)}\n"
            except TimeoutError:
                logger.warning("sync_indexing_timeout", collection_path=collection_path)
                result += "*Indexing timed out. Run `anki-atlas index` manually.*"
            except Exception as e:
                logger.exception(
                    "sync_indexing_failed",
                    collection_path=collection_path,
                    error_type=type(e).__name__,
                )
                result += f"*Indexing failed: {e}*"

        return result

    except Exception as e:
        logger.exception(
            "sync_failed", collection_path=collection_path, error_type=type(e).__name__
        )
        return _format_error(e, "sync")

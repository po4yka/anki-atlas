"""Topic coverage and gap analysis."""

from dataclasses import dataclass, field
from typing import Any

from packages.analytics.taxonomy import get_topic_by_path
from packages.common.config import Settings, get_settings
from packages.common.database import get_connection


@dataclass
class TopicCoverage:
    """Coverage metrics for a topic."""

    topic_id: int
    path: str
    label: str
    # Core metrics
    note_count: int = 0  # Notes labeled to this topic
    subtree_count: int = 0  # Notes in this topic + all children
    child_count: int = 0  # Number of child topics
    covered_children: int = 0  # Children with at least one note
    # Maturity metrics
    mature_count: int = 0  # Notes with mature cards
    avg_confidence: float = 0.0  # Average labeling confidence
    # Weakness metrics
    weak_notes: int = 0  # Notes with high lapse rate
    avg_lapses: float = 0.0


@dataclass
class TopicGap:
    """A gap (missing or undercovered topic)."""

    topic_id: int
    path: str
    label: str
    description: str | None
    gap_type: str  # "missing" or "undercovered"
    note_count: int
    threshold: int
    # Nearest notes that might cover this topic
    nearest_notes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class WeakNote:
    """A note with weakness signals."""

    note_id: int
    topic_path: str
    confidence: float
    lapses: int
    fail_rate: float | None
    normalized_text: str


async def get_topic_coverage(
    topic_path: str,
    include_subtree: bool = True,
    settings: Settings | None = None,
) -> TopicCoverage | None:
    """Get coverage metrics for a topic.

    Args:
        topic_path: Topic path to analyze.
        include_subtree: Include child topics in metrics.
        settings: Application settings.

    Returns:
        TopicCoverage or None if topic not found.
    """
    settings = settings or get_settings()

    # Get topic
    topic = await get_topic_by_path(topic_path, settings)
    if not topic or not topic.topic_id:
        return None

    coverage = TopicCoverage(
        topic_id=topic.topic_id,
        path=topic.path,
        label=topic.label,
    )

    async with get_connection(settings) as conn:
        # Get direct note count and metrics
        result = await conn.execute(
            """
            SELECT
                COUNT(DISTINCT nt.note_id) as note_count,
                AVG(nt.confidence) as avg_confidence,
                COUNT(DISTINCT CASE WHEN c.ivl >= 21 THEN nt.note_id END) as mature_count,
                AVG(c.lapses) as avg_lapses,
                COUNT(DISTINCT CASE WHEN cs.fail_rate > 0.3 THEN nt.note_id END) as weak_notes
            FROM note_topics nt
            JOIN topics t ON t.topic_id = nt.topic_id
            LEFT JOIN cards c ON c.note_id = nt.note_id
            LEFT JOIN card_stats cs ON cs.card_id = c.card_id
            WHERE t.path = %(path)s OR (%(include_subtree)s AND t.path LIKE %(path_prefix)s)
            """,
            {
                "path": topic_path,
                "include_subtree": include_subtree,
                "path_prefix": f"{topic_path}/%",
            },
        )
        row = await result.fetchone()
        if row:
            coverage.note_count = row["note_count"] or 0
            coverage.subtree_count = coverage.note_count
            coverage.avg_confidence = float(row["avg_confidence"] or 0)
            coverage.mature_count = row["mature_count"] or 0
            coverage.avg_lapses = float(row["avg_lapses"] or 0)
            coverage.weak_notes = row["weak_notes"] or 0

        # Get child topic counts
        result = await conn.execute(
            """
            SELECT
                COUNT(*) as child_count,
                COUNT(DISTINCT CASE WHEN EXISTS (
                    SELECT 1 FROM note_topics nt2
                    WHERE nt2.topic_id = t.topic_id
                ) THEN t.topic_id END) as covered_children
            FROM topics t
            WHERE t.path LIKE %(path_prefix)s
                AND t.path != %(path)s
                AND t.path NOT LIKE %(deeper_prefix)s
            """,
            {
                "path": topic_path,
                "path_prefix": f"{topic_path}/%",
                "deeper_prefix": f"{topic_path}/%/%",
            },
        )
        row = await result.fetchone()
        if row:
            coverage.child_count = row["child_count"] or 0
            coverage.covered_children = row["covered_children"] or 0

    return coverage


async def get_topic_gaps(
    topic_path: str,
    min_coverage: int = 1,
    settings: Settings | None = None,
) -> list[TopicGap]:
    """Find gaps in topic coverage.

    Args:
        topic_path: Root topic path to analyze.
        min_coverage: Minimum notes for a topic to be considered covered.
        settings: Application settings.

    Returns:
        List of gaps (missing or undercovered topics).
    """
    settings = settings or get_settings()
    gaps: list[TopicGap] = []

    async with get_connection(settings) as conn:
        # Find topics with insufficient coverage
        result = await conn.execute(
            """
            SELECT
                t.topic_id,
                t.path,
                t.label,
                t.description,
                COUNT(DISTINCT nt.note_id) as note_count
            FROM topics t
            LEFT JOIN note_topics nt ON nt.topic_id = t.topic_id
            WHERE t.path = %(path)s OR t.path LIKE %(path_prefix)s
            GROUP BY t.topic_id, t.path, t.label, t.description
            HAVING COUNT(DISTINCT nt.note_id) < %(min_coverage)s
            ORDER BY t.path
            """,
            {
                "path": topic_path,
                "path_prefix": f"{topic_path}/%",
                "min_coverage": min_coverage,
            },
        )

        async for row in result:
            gap_type = "missing" if row["note_count"] == 0 else "undercovered"
            gaps.append(
                TopicGap(
                    topic_id=row["topic_id"],
                    path=row["path"],
                    label=row["label"],
                    description=row["description"],
                    gap_type=gap_type,
                    note_count=row["note_count"],
                    threshold=min_coverage,
                )
            )

    return gaps


async def get_weak_notes(
    topic_path: str,
    max_results: int = 20,
    min_fail_rate: float = 0.2,
    settings: Settings | None = None,
) -> list[WeakNote]:
    """Get notes with weakness signals in a topic.

    Args:
        topic_path: Topic path to analyze.
        max_results: Maximum notes to return.
        min_fail_rate: Minimum fail rate to consider weak.
        settings: Application settings.

    Returns:
        List of weak notes.
    """
    settings = settings or get_settings()
    weak_notes: list[WeakNote] = []

    async with get_connection(settings) as conn:
        result = await conn.execute(
            """
            SELECT
                n.note_id,
                t.path as topic_path,
                nt.confidence,
                COALESCE(SUM(c.lapses), 0) as lapses,
                AVG(cs.fail_rate) as fail_rate,
                n.normalized_text
            FROM notes n
            JOIN note_topics nt ON nt.note_id = n.note_id
            JOIN topics t ON t.topic_id = nt.topic_id
            LEFT JOIN cards c ON c.note_id = n.note_id
            LEFT JOIN card_stats cs ON cs.card_id = c.card_id
            WHERE (t.path = %(path)s OR t.path LIKE %(path_prefix)s)
                AND n.deleted_at IS NULL
            GROUP BY n.note_id, t.path, nt.confidence, n.normalized_text
            HAVING AVG(cs.fail_rate) >= %(min_fail_rate)s OR SUM(c.lapses) > 5
            ORDER BY AVG(cs.fail_rate) DESC NULLS LAST, SUM(c.lapses) DESC
            LIMIT %(limit)s
            """,
            {
                "path": topic_path,
                "path_prefix": f"{topic_path}/%",
                "min_fail_rate": min_fail_rate,
                "limit": max_results,
            },
        )

        async for row in result:
            weak_notes.append(
                WeakNote(
                    note_id=row["note_id"],
                    topic_path=row["topic_path"],
                    confidence=row["confidence"],
                    lapses=row["lapses"],
                    fail_rate=row["fail_rate"],
                    normalized_text=row["normalized_text"][:200],  # Truncate
                )
            )

    return weak_notes


async def get_coverage_tree(
    root_path: str | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Get coverage tree for all topics.

    Args:
        root_path: Optional root path to filter.
        settings: Application settings.

    Returns:
        List of topic coverage data with hierarchy info.
    """
    settings = settings or get_settings()
    tree: list[dict[str, Any]] = []

    async with get_connection(settings) as conn:
        where_clause = ""
        params: dict[str, Any] = {}

        if root_path:
            where_clause = "WHERE t.path = %(path)s OR t.path LIKE %(path_prefix)s"
            params = {"path": root_path, "path_prefix": f"{root_path}/%"}

        result = await conn.execute(
            f"""
            SELECT
                t.topic_id,
                t.path,
                t.label,
                t.description,
                COUNT(DISTINCT nt.note_id) as note_count,
                AVG(nt.confidence) as avg_confidence,
                COUNT(DISTINCT CASE WHEN c.ivl >= 21 THEN nt.note_id END) as mature_count
            FROM topics t
            LEFT JOIN note_topics nt ON nt.topic_id = t.topic_id
            LEFT JOIN cards c ON c.note_id = nt.note_id
            {where_clause}
            GROUP BY t.topic_id, t.path, t.label, t.description
            ORDER BY t.path
            """,
            params,
        )

        async for row in result:
            tree.append({
                "topic_id": row["topic_id"],
                "path": row["path"],
                "label": row["label"],
                "description": row["description"],
                "note_count": row["note_count"] or 0,
                "avg_confidence": float(row["avg_confidence"] or 0),
                "mature_count": row["mature_count"] or 0,
                "depth": row["path"].count("/"),
            })

    return tree

"""Near-duplicate detection using embedding similarity."""

from dataclasses import dataclass, field
from typing import Any

from packages.common.config import Settings, get_settings
from packages.common.database import get_connection
from packages.indexer.qdrant import QdrantRepository, get_qdrant_repository


@dataclass
class DuplicatePair:
    """A pair of duplicate notes."""

    note_id_a: int
    note_id_b: int
    similarity: float


@dataclass
class DuplicateCluster:
    """A cluster of duplicate notes."""

    representative_id: int
    representative_text: str
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    # Metadata
    deck_names: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Total notes in cluster (representative + duplicates)."""
        return 1 + len(self.duplicates)


@dataclass
class DuplicateStats:
    """Statistics from duplicate detection."""

    notes_scanned: int = 0
    clusters_found: int = 0
    total_duplicates: int = 0
    avg_cluster_size: float = 0.0


class DuplicateDetector:
    """Service for detecting near-duplicate notes."""

    def __init__(
        self,
        settings: Settings | None = None,
        qdrant_repository: QdrantRepository | None = None,
    ) -> None:
        """Initialize detector.

        Args:
            settings: Application settings.
            qdrant_repository: Qdrant repository.
        """
        self.settings = settings or get_settings()
        self._qdrant_repository = qdrant_repository

    async def get_qdrant_repository(self) -> QdrantRepository:
        """Get or create Qdrant repository."""
        if self._qdrant_repository is None:
            self._qdrant_repository = await get_qdrant_repository(self.settings)
        return self._qdrant_repository

    async def find_duplicates(
        self,
        threshold: float = 0.92,
        max_clusters: int = 100,
        deck_filter: list[str] | None = None,
        tag_filter: list[str] | None = None,
    ) -> tuple[list[DuplicateCluster], DuplicateStats]:
        """Find clusters of near-duplicate notes.

        Args:
            threshold: Minimum similarity threshold (0-1).
            max_clusters: Maximum clusters to return.
            deck_filter: Optional deck name filter.
            tag_filter: Optional tag filter.

        Returns:
            Tuple of (clusters, statistics).
        """
        stats = DuplicateStats()
        qdrant = await self.get_qdrant_repository()

        # Get all note IDs and their vectors
        note_ids = await self._get_note_ids(deck_filter, tag_filter)
        stats.notes_scanned = len(note_ids)

        if not note_ids:
            return [], stats

        # Find duplicate pairs
        pairs = await self._find_duplicate_pairs(
            qdrant, note_ids, threshold, deck_filter, tag_filter
        )

        # Cluster duplicates
        clusters = self._cluster_duplicates(pairs)

        # Enrich with note details
        enriched_clusters = await self._enrich_clusters(clusters)

        # Sort by cluster size (largest first)
        enriched_clusters.sort(key=lambda c: c.size, reverse=True)

        # Limit clusters
        result_clusters = enriched_clusters[:max_clusters]

        # Calculate stats
        stats.clusters_found = len(result_clusters)
        stats.total_duplicates = sum(len(c.duplicates) for c in result_clusters)
        if result_clusters:
            stats.avg_cluster_size = sum(c.size for c in result_clusters) / len(result_clusters)

        return result_clusters, stats

    async def _get_note_ids(
        self,
        deck_filter: list[str] | None,
        tag_filter: list[str] | None,
    ) -> list[int]:
        """Get note IDs matching filters."""
        async with get_connection(self.settings) as conn:
            where_clauses = ["n.deleted_at IS NULL"]
            params: dict[str, Any] = {}

            if tag_filter:
                where_clauses.append("n.tags && %(tags)s")
                params["tags"] = tag_filter

            if deck_filter:
                where_clauses.append(
                    """
                    EXISTS (
                        SELECT 1 FROM cards c
                        JOIN decks d ON d.deck_id = c.deck_id
                        WHERE c.note_id = n.note_id AND d.name = ANY(%(deck_names)s)
                    )
                    """
                )
                params["deck_names"] = deck_filter

            where_sql = " AND ".join(where_clauses)

            result = await conn.execute(
                f"SELECT note_id FROM notes n WHERE {where_sql} ORDER BY note_id",
                params,
            )

            return [row["note_id"] async for row in result]

    async def _find_duplicate_pairs(
        self,
        qdrant: QdrantRepository,
        note_ids: list[int],
        threshold: float,
        deck_filter: list[str] | None,
        tag_filter: list[str] | None,
    ) -> list[DuplicatePair]:
        """Find all pairs of duplicates above threshold."""
        pairs: list[DuplicatePair] = []
        seen_pairs: set[tuple[int, int]] = set()

        # Process in batches to avoid memory issues
        batch_size = 100

        for i in range(0, len(note_ids), batch_size):
            batch_ids = note_ids[i : i + batch_size]

            for note_id in batch_ids:
                # Find similar notes
                similar = await qdrant.find_similar_to_note(
                    note_id=note_id,
                    limit=10,
                    min_score=threshold,
                    deck_names=deck_filter,
                    tags=tag_filter,
                )

                for similar_id, score in similar:
                    if similar_id == note_id:
                        continue

                    # Create ordered pair to avoid duplicates
                    pair_key = (min(note_id, similar_id), max(note_id, similar_id))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        pairs.append(
                            DuplicatePair(
                                note_id_a=pair_key[0],
                                note_id_b=pair_key[1],
                                similarity=score,
                            )
                        )

        return pairs

    def _cluster_duplicates(
        self,
        pairs: list[DuplicatePair],
    ) -> dict[int, list[tuple[int, float]]]:
        """Cluster duplicate pairs using union-find."""
        if not pairs:
            return {}

        # Union-find for clustering
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                # Always use smaller ID as parent (deterministic)
                if px < py:
                    parent[py] = px
                else:
                    parent[px] = py

        # Build clusters
        for pair in pairs:
            union(pair.note_id_a, pair.note_id_b)

        # Group by cluster root
        clusters: dict[int, list[tuple[int, float]]] = {}
        pair_scores: dict[tuple[int, int], float] = {
            (p.note_id_a, p.note_id_b): p.similarity for p in pairs
        }

        all_ids = set()
        for pair in pairs:
            all_ids.add(pair.note_id_a)
            all_ids.add(pair.note_id_b)

        for note_id in all_ids:
            root = find(note_id)
            if root not in clusters:
                clusters[root] = []
            if note_id != root:
                # Find similarity to root
                pair_key = (min(note_id, root), max(note_id, root))
                sim = pair_scores.get(pair_key, 0.0)
                # If no direct pair, find via any connected pair
                if sim == 0.0:
                    for pair in pairs:
                        if note_id in (pair.note_id_a, pair.note_id_b):
                            sim = max(sim, pair.similarity)
                clusters[root].append((note_id, sim))

        return clusters

    async def _enrich_clusters(
        self,
        clusters: dict[int, list[tuple[int, float]]],
    ) -> list[DuplicateCluster]:
        """Enrich clusters with note details."""
        if not clusters:
            return []

        # Get all note IDs
        all_ids = set(clusters.keys())
        for members in clusters.values():
            for note_id, _ in members:
                all_ids.add(note_id)

        # Fetch note details
        note_details = await self._get_note_details(list(all_ids))

        # Build enriched clusters
        result: list[DuplicateCluster] = []

        for rep_id, members in clusters.items():
            rep_detail = note_details.get(rep_id, {})

            cluster = DuplicateCluster(
                representative_id=rep_id,
                representative_text=rep_detail.get("text", "")[:200],
                deck_names=rep_detail.get("deck_names", []),
                tags=rep_detail.get("tags", []),
            )

            for note_id, similarity in members:
                detail = note_details.get(note_id, {})
                cluster.duplicates.append(
                    {
                        "note_id": note_id,
                        "similarity": similarity,
                        "text": detail.get("text", "")[:200],
                        "deck_names": detail.get("deck_names", []),
                        "tags": detail.get("tags", []),
                    }
                )

            # Sort duplicates by similarity
            cluster.duplicates.sort(key=lambda d: d["similarity"], reverse=True)
            result.append(cluster)

        return result

    async def _get_note_details(
        self,
        note_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        """Fetch note details for display."""
        if not note_ids:
            return {}

        async with get_connection(self.settings) as conn:
            result = await conn.execute(
                """
                SELECT
                    n.note_id,
                    n.normalized_text,
                    n.tags,
                    COALESCE(
                        array_agg(DISTINCT d.name) FILTER (WHERE d.name IS NOT NULL),
                        '{}'::text[]
                    ) as deck_names
                FROM notes n
                LEFT JOIN cards c ON c.note_id = n.note_id
                LEFT JOIN decks d ON d.deck_id = c.deck_id
                WHERE n.note_id = ANY(%(note_ids)s)
                GROUP BY n.note_id, n.normalized_text, n.tags
                """,
                {"note_ids": note_ids},
            )

            details: dict[int, dict[str, Any]] = {}
            async for row in result:
                details[row["note_id"]] = {
                    "text": row["normalized_text"],
                    "tags": row["tags"] or [],
                    "deck_names": row["deck_names"] or [],
                }

            return details


async def find_duplicates(
    threshold: float = 0.92,
    max_clusters: int = 100,
    settings: Settings | None = None,
) -> tuple[list[DuplicateCluster], DuplicateStats]:
    """Convenience function to find duplicates.

    Args:
        threshold: Similarity threshold.
        max_clusters: Maximum clusters.
        settings: Application settings.

    Returns:
        Tuple of (clusters, statistics).
    """
    detector = DuplicateDetector(settings)
    return await detector.find_duplicates(threshold, max_clusters)

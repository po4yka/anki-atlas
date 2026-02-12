"""Index service for embedding and storing note vectors."""

import json
from dataclasses import dataclass, field
from typing import Any

from psycopg import AsyncConnection

from packages.common.config import Settings, get_settings
from packages.common.database import get_connection
from packages.common.logging import get_logger
from packages.indexer.embeddings import EmbeddingProvider, get_embedding_provider
from packages.indexer.qdrant import NotePayload, QdrantRepository, get_qdrant_repository

logger = get_logger(module=__name__)


class EmbeddingModelChanged(Exception):
    """Raised when embedding model changed since last indexing."""

    def __init__(self, stored: str, current: str) -> None:
        self.stored_version = stored
        self.current_version = current
        super().__init__(
            f"Embedding model changed: '{stored}' -> '{current}'. "
            f"Use --force-reindex to re-embed all notes with the new model."
        )


@dataclass
class NoteForIndexing:
    """Note data needed for indexing."""

    note_id: int
    model_id: int
    normalized_text: str
    tags: list[str]
    deck_names: list[str]
    # Card stats (aggregated)
    mature: bool = False
    lapses: int = 0
    reps: int = 0
    fail_rate: float | None = None


@dataclass
class IndexStats:
    """Statistics from an indexing operation."""

    notes_processed: int = 0
    notes_embedded: int = 0
    notes_skipped: int = 0
    notes_deleted: int = 0
    errors: list[str] = field(default_factory=list)


class IndexService:
    """Service for indexing notes to vector database."""

    # Version for tracking changes that require re-indexing
    NORMALIZATION_VERSION = "1"

    def __init__(
        self,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        qdrant_repository: QdrantRepository | None = None,
    ) -> None:
        """Initialize index service.

        Args:
            settings: Application settings.
            embedding_provider: Embedding provider (uses default if None).
            qdrant_repository: Qdrant repository (uses default if None).
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

    def _embedding_version(self, provider: EmbeddingProvider) -> str:
        """Get current embedding version string."""
        return f"{self.NORMALIZATION_VERSION}:{provider.model_name}:{provider.dimension}"

    async def index_notes(
        self,
        notes: list[NoteForIndexing],
        force_reindex: bool = False,
    ) -> IndexStats:
        """Index notes to vector database.

        Args:
            notes: Notes to index.
            force_reindex: If True, re-embed all notes regardless of hash.

        Returns:
            IndexStats with operation counts.
        """
        stats = IndexStats(notes_processed=len(notes))

        if not notes:
            return stats

        provider = await self.get_embedding_provider()
        qdrant = await self.get_qdrant_repository()

        # Ensure collection exists
        await qdrant.ensure_collection(provider.dimension)

        # Get existing hashes to check what needs updating
        note_ids = [n.note_id for n in notes]
        existing_hashes: dict[int, str] = {}
        if not force_reindex:
            existing_hashes = await qdrant.get_existing_hashes(note_ids)

        # Determine which notes need embedding
        notes_to_embed: list[NoteForIndexing] = []
        content_hashes: dict[int, str] = {}

        for note in notes:
            content_hash = provider.content_hash(note.normalized_text)
            content_hashes[note.note_id] = content_hash

            if force_reindex or existing_hashes.get(note.note_id) != content_hash:
                notes_to_embed.append(note)
            else:
                stats.notes_skipped += 1

        if not notes_to_embed:
            return stats

        # Embed texts
        texts = [n.normalized_text for n in notes_to_embed]
        try:
            vectors = await provider.embed(texts)
        except Exception as e:
            logger.exception("embedding_failed", batch_size=len(texts))
            stats.errors.append(f"Embedding failed: {e}")
            return stats

        # Build payloads
        payloads = [
            NotePayload(
                note_id=note.note_id,
                deck_names=note.deck_names,
                tags=note.tags,
                model_id=note.model_id,
                content_hash=content_hashes[note.note_id],
                mature=note.mature,
                lapses=note.lapses,
                reps=note.reps,
                fail_rate=note.fail_rate,
            )
            for note in notes_to_embed
        ]

        # Upsert to Qdrant
        try:
            upserted = await qdrant.upsert_vectors(vectors, payloads)
            stats.notes_embedded = upserted
        except Exception as e:
            logger.exception("qdrant_upsert_failed", batch_size=len(vectors))
            stats.errors.append(f"Qdrant upsert failed: {e}")

        return stats

    async def delete_notes(self, note_ids: list[int]) -> int:
        """Delete notes from vector database.

        Args:
            note_ids: Note IDs to delete.

        Returns:
            Number of notes deleted.
        """
        if not note_ids:
            return 0

        qdrant = await self.get_qdrant_repository()
        return await qdrant.delete_vectors(note_ids)

    async def _get_stored_embedding_version(self) -> str | None:
        """Read embedding_version from sync_metadata table.

        Returns:
            The stored version string, or None if not yet set.
        """
        async with get_connection(self.settings) as conn:
            result = await conn.execute(
                "SELECT value FROM sync_metadata WHERE key = 'embedding_version'"
            )
            row = await result.fetchone()
            if row is None:
                return None
            value: str = json.loads(row["value"])
            return value

    async def index_from_database(
        self,
        force_reindex: bool = False,
        batch_size: int = 500,
    ) -> IndexStats:
        """Index all notes from PostgreSQL database.

        Args:
            force_reindex: If True, re-embed all notes.
            batch_size: Number of notes to process per batch.

        Returns:
            Combined IndexStats from all batches.

        Raises:
            EmbeddingModelChanged: If the embedding model changed and
                force_reindex is False.
        """
        total_stats = IndexStats()

        provider = await self.get_embedding_provider()
        current_version = self._embedding_version(provider)
        stored_version = await self._get_stored_embedding_version()

        if stored_version is not None and stored_version != current_version:
            if not force_reindex:
                raise EmbeddingModelChanged(stored_version, current_version)
            # Force reindex: recreate the collection with the new dimension
            qdrant = await self.get_qdrant_repository()
            await qdrant.recreate_collection(provider.dimension)
            logger.warning(
                "embedding_model_changed_recreating",
                stored=stored_version,
                current=current_version,
            )

        async with get_connection(self.settings) as conn:
            # Get total count
            result = await conn.execute(
                "SELECT COUNT(*) as count FROM notes WHERE deleted_at IS NULL"
            )
            row = await result.fetchone()
            total_count = row["count"] if row else 0

            # Process in batches
            offset = 0
            while offset < total_count:
                notes = await self._fetch_notes_batch(conn, offset, batch_size)
                if not notes:
                    break

                batch_stats = await self.index_notes(notes, force_reindex)
                total_stats.notes_processed += batch_stats.notes_processed
                total_stats.notes_embedded += batch_stats.notes_embedded
                total_stats.notes_skipped += batch_stats.notes_skipped
                total_stats.errors.extend(batch_stats.errors)

                offset += batch_size

            # Handle deleted notes
            deleted_ids = await self._get_deleted_note_ids(conn)
            if deleted_ids:
                deleted_count = await self.delete_notes(deleted_ids)
                total_stats.notes_deleted = deleted_count

        # Update sync metadata
        await self._update_index_metadata()

        return total_stats

    async def _fetch_notes_batch(
        self,
        conn: AsyncConnection[dict[str, Any]],
        offset: int,
        limit: int,
    ) -> list[NoteForIndexing]:
        """Fetch a batch of notes with their metadata."""
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
                COALESCE(SUM(c.reps), 0) as reps,
                COALESCE(AVG(cs.fail_rate), NULL) as fail_rate
            FROM notes n
            LEFT JOIN cards c ON c.note_id = n.note_id
            LEFT JOIN decks d ON d.deck_id = c.deck_id
            LEFT JOIN card_stats cs ON cs.card_id = c.card_id
            WHERE n.deleted_at IS NULL
            GROUP BY n.note_id, n.model_id, n.normalized_text, n.tags
            ORDER BY n.note_id
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        )

        notes: list[NoteForIndexing] = []
        async for row in result:
            notes.append(
                NoteForIndexing(
                    note_id=row["note_id"],
                    model_id=row["model_id"],
                    normalized_text=row["normalized_text"],
                    tags=row["tags"] or [],
                    deck_names=row["deck_names"] or [],
                    mature=row["mature"],
                    lapses=row["lapses"],
                    reps=row["reps"],
                    fail_rate=row["fail_rate"],
                )
            )

        return notes

    async def _get_deleted_note_ids(
        self,
        conn: AsyncConnection[dict[str, Any]],
    ) -> list[int]:
        """Get IDs of deleted notes that may still be in Qdrant."""
        result = await conn.execute("SELECT note_id FROM notes WHERE deleted_at IS NOT NULL")
        return [row["note_id"] async for row in result]

    async def _update_index_metadata(self) -> None:
        """Update index metadata in sync_metadata table."""
        provider = await self.get_embedding_provider()
        version = self._embedding_version(provider)

        async with get_connection(self.settings) as conn:
            await conn.execute(
                """
                INSERT INTO sync_metadata (key, value)
                VALUES ('embedding_version', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (json.dumps(version),),
            )
            await conn.commit()


async def index_all_notes(
    settings: Settings | None = None,
    force_reindex: bool = False,
) -> IndexStats:
    """Convenience function to index all notes.

    Args:
        settings: Application settings.
        force_reindex: If True, re-embed all notes.

    Returns:
        IndexStats with operation counts.
    """
    service = IndexService(settings)
    return await service.index_from_database(force_reindex)

"""Sync service for Anki collection to PostgreSQL."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from psycopg import AsyncConnection

from packages.anki.models import AnkiCollection
from packages.anki.normalizer import build_card_deck_map, build_deck_map, normalize_notes
from packages.anki.reader import read_anki_collection
from packages.common.config import Settings, get_settings
from packages.common.database import get_connection


@dataclass
class SyncStats:
    """Statistics from a sync operation."""

    decks_upserted: int = 0
    models_upserted: int = 0
    notes_upserted: int = 0
    notes_deleted: int = 0
    cards_upserted: int = 0
    card_stats_upserted: int = 0
    duration_ms: int = 0


class SyncService:
    """Service for syncing Anki collections to PostgreSQL."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize sync service."""
        self.settings = settings or get_settings()

    async def sync_collection(
        self,
        collection_path: str | Path,
    ) -> SyncStats:
        """Sync an Anki collection to PostgreSQL.

        Args:
            collection_path: Path to collection.anki2 file.

        Returns:
            SyncStats with counts of upserted/deleted records.
        """
        start_time = datetime.now(UTC)
        stats = SyncStats()

        # Read collection from SQLite
        collection = read_anki_collection(collection_path)

        # Normalize notes
        deck_map = build_deck_map(collection.decks)
        card_deck_map = build_card_deck_map(collection.cards)
        normalize_notes(collection.notes, deck_map, card_deck_map)

        # Sync to PostgreSQL
        async with get_connection(self.settings) as conn:
            stats.decks_upserted = await self._sync_decks(conn, collection)
            stats.models_upserted = await self._sync_models(conn, collection)
            stats.notes_upserted = await self._sync_notes(conn, collection)
            stats.notes_deleted = await self._delete_missing_notes(conn, collection)
            stats.cards_upserted = await self._sync_cards(conn, collection)
            stats.card_stats_upserted = await self._sync_card_stats(conn, collection)

            # Update sync metadata
            await self._update_sync_metadata(conn, collection)

            await conn.commit()

        end_time = datetime.now(UTC)
        stats.duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return stats

    async def _sync_decks(
        self,
        conn: AsyncConnection[dict[str, Any]],
        collection: AnkiCollection,
    ) -> int:
        """Sync decks to PostgreSQL."""
        count = 0
        for deck in collection.decks:
            await conn.execute(
                """
                INSERT INTO decks (deck_id, name, parent_name, config)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (deck_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    parent_name = EXCLUDED.parent_name,
                    config = EXCLUDED.config
                """,
                (
                    deck.deck_id,
                    deck.name,
                    deck.parent_name,
                    json.dumps(deck.config),
                ),
            )
            count += 1
        return count

    async def _sync_models(
        self,
        conn: AsyncConnection[dict[str, Any]],
        collection: AnkiCollection,
    ) -> int:
        """Sync note models to PostgreSQL."""
        count = 0
        for model in collection.models:
            await conn.execute(
                """
                INSERT INTO models (model_id, name, fields, templates, config)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (model_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    fields = EXCLUDED.fields,
                    templates = EXCLUDED.templates,
                    config = EXCLUDED.config
                """,
                (
                    model.model_id,
                    model.name,
                    json.dumps(model.fields),
                    json.dumps(model.templates),
                    json.dumps(model.config),
                ),
            )
            count += 1
        return count

    async def _sync_notes(
        self,
        conn: AsyncConnection[dict[str, Any]],
        collection: AnkiCollection,
    ) -> int:
        """Sync notes to PostgreSQL."""
        count = 0
        for note in collection.notes:
            await conn.execute(
                """
                INSERT INTO notes (
                    note_id, model_id, tags, fields_json, raw_fields,
                    normalized_text, mtime, usn, deleted_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL)
                ON CONFLICT (note_id) DO UPDATE SET
                    model_id = EXCLUDED.model_id,
                    tags = EXCLUDED.tags,
                    fields_json = EXCLUDED.fields_json,
                    raw_fields = EXCLUDED.raw_fields,
                    normalized_text = EXCLUDED.normalized_text,
                    mtime = EXCLUDED.mtime,
                    usn = EXCLUDED.usn,
                    deleted_at = NULL
                """,
                (
                    note.note_id,
                    note.model_id,
                    note.tags,
                    json.dumps(note.fields_json),
                    note.raw_fields,
                    note.normalized_text,
                    note.mtime,
                    note.usn,
                ),
            )
            count += 1
        return count

    async def _delete_missing_notes(
        self,
        conn: AsyncConnection[dict[str, Any]],
        collection: AnkiCollection,
    ) -> int:
        """Soft-delete notes that are no longer in the source collection."""
        if not collection.notes:
            return 0

        note_ids = [note.note_id for note in collection.notes]

        # Soft-delete notes not in the current collection
        result = await conn.execute(
            """
            UPDATE notes
            SET deleted_at = NOW()
            WHERE note_id NOT IN (SELECT unnest(%s::bigint[]))
              AND deleted_at IS NULL
            """,
            (note_ids,),
        )

        return result.rowcount or 0

    async def _sync_cards(
        self,
        conn: AsyncConnection[dict[str, Any]],
        collection: AnkiCollection,
    ) -> int:
        """Sync cards to PostgreSQL."""
        count = 0
        for card in collection.cards:
            await conn.execute(
                """
                INSERT INTO cards (
                    card_id, note_id, deck_id, ord, due, ivl, ease,
                    lapses, reps, queue, type, mtime, usn
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (card_id) DO UPDATE SET
                    note_id = EXCLUDED.note_id,
                    deck_id = EXCLUDED.deck_id,
                    ord = EXCLUDED.ord,
                    due = EXCLUDED.due,
                    ivl = EXCLUDED.ivl,
                    ease = EXCLUDED.ease,
                    lapses = EXCLUDED.lapses,
                    reps = EXCLUDED.reps,
                    queue = EXCLUDED.queue,
                    type = EXCLUDED.type,
                    mtime = EXCLUDED.mtime,
                    usn = EXCLUDED.usn
                """,
                (
                    card.card_id,
                    card.note_id,
                    card.deck_id,
                    card.ord,
                    card.due,
                    card.ivl,
                    card.ease,
                    card.lapses,
                    card.reps,
                    card.queue,
                    card.type,
                    card.mtime,
                    card.usn,
                ),
            )
            count += 1
        return count

    async def _sync_card_stats(
        self,
        conn: AsyncConnection[dict[str, Any]],
        collection: AnkiCollection,
    ) -> int:
        """Sync card stats to PostgreSQL."""
        # Filter out stats for cards that no longer exist (orphaned revlog entries)
        valid_card_ids = {card.card_id for card in collection.cards}
        count = 0
        for stats in collection.card_stats:
            if stats.card_id not in valid_card_ids:
                continue
            await conn.execute(
                """
                INSERT INTO card_stats (
                    card_id, reviews, avg_ease, fail_rate,
                    last_review_at, total_time_ms
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (card_id) DO UPDATE SET
                    reviews = EXCLUDED.reviews,
                    avg_ease = EXCLUDED.avg_ease,
                    fail_rate = EXCLUDED.fail_rate,
                    last_review_at = EXCLUDED.last_review_at,
                    total_time_ms = EXCLUDED.total_time_ms
                """,
                (
                    stats.card_id,
                    stats.reviews,
                    stats.avg_ease,
                    stats.fail_rate,
                    stats.last_review_at,
                    stats.total_time_ms,
                ),
            )
            count += 1
        return count

    async def _update_sync_metadata(
        self,
        conn: AsyncConnection[dict[str, Any]],
        collection: AnkiCollection,
    ) -> None:
        """Update sync metadata."""
        now = datetime.now(UTC).isoformat()
        await conn.execute(
            """
            INSERT INTO sync_metadata (key, value)
            VALUES ('last_sync_at', %s)
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value
            """,
            (json.dumps(now),),
        )

        await conn.execute(
            """
            INSERT INTO sync_metadata (key, value)
            VALUES ('last_collection_path', %s)
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value
            """,
            (json.dumps(collection.collection_path),),
        )


async def sync_anki_collection(
    collection_path: str | Path,
    settings: Settings | None = None,
) -> SyncStats:
    """Convenience function to sync an Anki collection.

    Args:
        collection_path: Path to collection.anki2 file.
        settings: Optional settings override.

    Returns:
        SyncStats with operation counts.
    """
    service = SyncService(settings)
    return await service.sync_collection(collection_path)

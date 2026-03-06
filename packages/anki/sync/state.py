"""SQLite WAL state database for tracking sync state."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class CardState:
    """Tracked state for a single card."""

    slug: str
    content_hash: str
    anki_guid: int | None = None
    note_type: str = ""
    source_path: str = ""
    synced_at: float = 0.0


class StateDB:
    """SQLite WAL database for tracking sync state."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()
        log.debug("state_db.opened", path=str(db_path))

    def _create_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS card_state (
                slug TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                anki_guid INTEGER,
                note_type TEXT NOT NULL DEFAULT '',
                source_path TEXT NOT NULL DEFAULT '',
                synced_at REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        self._conn.commit()

    def get(self, slug: str) -> CardState | None:
        """Get card state by slug."""
        row = self._conn.execute(
            "SELECT slug, content_hash, anki_guid, note_type, source_path, synced_at "
            "FROM card_state WHERE slug = ?",
            (slug,),
        ).fetchone()
        if row is None:
            return None
        return CardState(
            slug=row[0],
            content_hash=row[1],
            anki_guid=row[2],
            note_type=row[3],
            source_path=row[4],
            synced_at=row[5],
        )

    def get_all(self) -> tuple[CardState, ...]:
        """Get all card states."""
        rows = self._conn.execute(
            "SELECT slug, content_hash, anki_guid, note_type, source_path, synced_at "
            "FROM card_state ORDER BY slug"
        ).fetchall()
        return tuple(
            CardState(
                slug=r[0], content_hash=r[1], anki_guid=r[2],
                note_type=r[3], source_path=r[4], synced_at=r[5],
            )
            for r in rows
        )

    def upsert(self, state: CardState) -> None:
        """Insert or update card state."""
        self._conn.execute(
            """
            INSERT INTO card_state (slug, content_hash, anki_guid, note_type, source_path, synced_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (slug) DO UPDATE SET
                content_hash = excluded.content_hash,
                anki_guid = excluded.anki_guid,
                note_type = excluded.note_type,
                source_path = excluded.source_path,
                synced_at = excluded.synced_at
            """,
            (
                state.slug,
                state.content_hash,
                state.anki_guid,
                state.note_type,
                state.source_path,
                state.synced_at,
            ),
        )
        self._conn.commit()

    def delete(self, slug: str) -> None:
        """Delete card state by slug."""
        self._conn.execute("DELETE FROM card_state WHERE slug = ?", (slug,))
        self._conn.commit()

    def get_by_source(self, source_path: str) -> tuple[CardState, ...]:
        """Get all card states for a source path."""
        rows = self._conn.execute(
            "SELECT slug, content_hash, anki_guid, note_type, source_path, synced_at "
            "FROM card_state WHERE source_path = ? ORDER BY slug",
            (source_path,),
        ).fetchall()
        return tuple(
            CardState(
                slug=r[0], content_hash=r[1], anki_guid=r[2],
                note_type=r[3], source_path=r[4], synced_at=r[5],
            )
            for r in rows
        )

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()
        log.debug("state_db.closed", path=str(self._db_path))

    def __enter__(self) -> StateDB:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


def _now() -> float:
    """Current time as float (for synced_at)."""
    return time.time()

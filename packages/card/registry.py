"""SQLite-based card registry for tracking cards and notes."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

import structlog

from packages.card.slug import SlugService
from packages.common.exceptions import DatabaseError, MigrationError

log = structlog.get_logger()

SCHEMA_VERSION: Final[int] = 2

_CREATE_CARDS_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS cards (
    slug TEXT PRIMARY KEY,
    note_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    front TEXT NOT NULL,
    back TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata_hash TEXT NOT NULL,
    language TEXT NOT NULL,
    tags TEXT,
    anki_note_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    synced_at TEXT
)
"""

_CREATE_NOTES_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS notes (
    note_id TEXT PRIMARY KEY,
    source_path TEXT UNIQUE NOT NULL,
    title TEXT,
    content_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

_CREATE_SCHEMA_VERSION_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS schema_version (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL DEFAULT 1
)
"""

_CARD_INDEXES: Final[tuple[str, ...]] = (
    "CREATE INDEX IF NOT EXISTS idx_cards_content_hash ON cards (content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_cards_note_id ON cards (note_id)",
    "CREATE INDEX IF NOT EXISTS idx_cards_source_path ON cards (source_path)",
    "CREATE INDEX IF NOT EXISTS idx_cards_anki_note_id ON cards (anki_note_id)",
)

_NOTE_INDEXES: Final[tuple[str, ...]] = (
    "CREATE INDEX IF NOT EXISTS idx_notes_source_path ON notes (source_path)",
)


def compute_content_hash(front: str, back: str) -> str:
    """Compute 12-character content hash from front and back."""
    return SlugService.compute_content_hash(front, back)


def compute_metadata_hash(note_type: str, tags: Sequence[str]) -> str:
    """Compute 6-character metadata hash from note type and tags."""
    return SlugService.compute_metadata_hash(note_type, list(tags))


@dataclass(frozen=True, slots=True)
class CardEntry:
    """Registry entry for a tracked card."""

    slug: str
    note_id: str
    source_path: str
    front: str
    back: str
    content_hash: str
    metadata_hash: str
    language: str
    tags: tuple[str, ...] = ()
    anki_note_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    synced_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class NoteEntry:
    """Registry entry for a tracked note."""

    note_id: str
    source_path: str
    title: str | None = None
    content_hash: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _dt_to_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _tags_to_str(tags: tuple[str, ...]) -> str:
    return ",".join(tags)


def _str_to_tags(s: str | None) -> tuple[str, ...]:
    if not s:
        return ()
    return tuple(s.split(","))


class CardRegistry:
    """SQLite-backed registry for cards and notes."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = str(db_path) if db_path is not None else ":memory:"
        self._conn: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(self._db_path)
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
            except sqlite3.Error as e:
                raise DatabaseError(
                    f"Failed to connect to database: {e}",
                    context={"db_path": self._db_path},
                ) from e
            self._ensure_schema(self._conn)
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        if cursor.fetchone() is None:
            self._create_full_schema(conn)
        else:
            self._migrate_schema(conn)

    def _create_full_schema(self, conn: sqlite3.Connection) -> None:
        try:
            conn.execute(_CREATE_CARDS_TABLE)
            conn.execute(_CREATE_NOTES_TABLE)
            conn.execute(_CREATE_SCHEMA_VERSION_TABLE)
            for idx in _CARD_INDEXES:
                conn.execute(idx)
            for idx in _NOTE_INDEXES:
                conn.execute(idx)
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (id, version) VALUES (1, ?)",
                (SCHEMA_VERSION,),
            )
            conn.commit()
            log.info("card_registry.schema_created", version=SCHEMA_VERSION)
        except sqlite3.Error as e:
            raise MigrationError(
                f"Failed to create schema: {e}",
                context={"version": SCHEMA_VERSION},
            ) from e

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
        current = row["version"] if row else 1

        if current >= SCHEMA_VERSION:
            return

        try:
            if current < 2:
                conn.execute(_CREATE_NOTES_TABLE)
                for idx in _NOTE_INDEXES:
                    conn.execute(idx)
                log.info("card_registry.migrated", from_version=current, to_version=2)

            conn.execute(
                "UPDATE schema_version SET version = ? WHERE id = 1",
                (SCHEMA_VERSION,),
            )
            conn.commit()
        except sqlite3.Error as e:
            raise MigrationError(
                f"Schema migration failed: {e}",
                context={"from_version": current, "to_version": SCHEMA_VERSION},
            ) from e

    def _row_to_card_entry(self, row: sqlite3.Row) -> CardEntry:
        return CardEntry(
            slug=row["slug"],
            note_id=row["note_id"],
            source_path=row["source_path"],
            front=row["front"],
            back=row["back"],
            content_hash=row["content_hash"],
            metadata_hash=row["metadata_hash"],
            language=row["language"],
            tags=_str_to_tags(row["tags"]),
            anki_note_id=row["anki_note_id"],
            created_at=_str_to_dt(row["created_at"]),
            updated_at=_str_to_dt(row["updated_at"]),
            synced_at=_str_to_dt(row["synced_at"]),
        )

    def _row_to_note_entry(self, row: sqlite3.Row) -> NoteEntry:
        return NoteEntry(
            note_id=row["note_id"],
            source_path=row["source_path"],
            title=row["title"],
            content_hash=row["content_hash"],
            created_at=_str_to_dt(row["created_at"]),
            updated_at=_str_to_dt(row["updated_at"]),
        )

    # --- Card CRUD ---

    def add_card(self, entry: CardEntry) -> bool:
        """Insert a card entry. Returns True on success, False if slug exists."""
        conn = self._get_connection()
        now = _now_utc()
        created = entry.created_at or now
        updated = entry.updated_at or now
        try:
            conn.execute(
                """INSERT INTO cards
                   (slug, note_id, source_path, front, back, content_hash,
                    metadata_hash, language, tags, anki_note_id,
                    created_at, updated_at, synced_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.slug,
                    entry.note_id,
                    entry.source_path,
                    entry.front,
                    entry.back,
                    entry.content_hash,
                    entry.metadata_hash,
                    entry.language,
                    _tags_to_str(entry.tags),
                    entry.anki_note_id,
                    _dt_to_str(created),
                    _dt_to_str(updated),
                    _dt_to_str(entry.synced_at),
                ),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_card(self, slug: str) -> CardEntry | None:
        """Get a card by slug."""
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM cards WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            return None
        return self._row_to_card_entry(row)

    def update_card(self, entry: CardEntry) -> bool:
        """Update a card by slug. Returns True if updated, False if not found."""
        conn = self._get_connection()
        updated = entry.updated_at or _now_utc()
        cursor = conn.execute(
            """UPDATE cards SET
               note_id = ?, source_path = ?, front = ?, back = ?,
               content_hash = ?, metadata_hash = ?, language = ?,
               tags = ?, anki_note_id = ?, updated_at = ?, synced_at = ?
               WHERE slug = ?""",
            (
                entry.note_id,
                entry.source_path,
                entry.front,
                entry.back,
                entry.content_hash,
                entry.metadata_hash,
                entry.language,
                _tags_to_str(entry.tags),
                entry.anki_note_id,
                _dt_to_str(updated),
                _dt_to_str(entry.synced_at),
                entry.slug,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete_card(self, slug: str) -> bool:
        """Delete a card by slug. Returns True if deleted."""
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM cards WHERE slug = ?", (slug,))
        conn.commit()
        return cursor.rowcount > 0

    def find_cards(
        self,
        *,
        note_id: str | None = None,
        source_path: str | None = None,
        content_hash: str | None = None,
    ) -> list[CardEntry]:
        """Find cards by optional filters."""
        conn = self._get_connection()
        conditions: list[str] = []
        params: list[str] = []

        if note_id is not None:
            conditions.append("note_id = ?")
            params.append(note_id)
        if source_path is not None:
            conditions.append("source_path = ?")
            params.append(source_path)
        if content_hash is not None:
            conditions.append("content_hash = ?")
            params.append(content_hash)

        where = " AND ".join(conditions) if conditions else "1=1"
        rows = conn.execute(f"SELECT * FROM cards WHERE {where}", params).fetchall()
        return [self._row_to_card_entry(row) for row in rows]

    # --- Note CRUD ---

    def add_note(self, entry: NoteEntry) -> bool:
        """Insert a note entry. Returns True on success, False if note_id exists."""
        conn = self._get_connection()
        now = _now_utc()
        created = entry.created_at or now
        updated = entry.updated_at or now
        try:
            conn.execute(
                """INSERT INTO notes
                   (note_id, source_path, title, content_hash, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    entry.note_id,
                    entry.source_path,
                    entry.title,
                    entry.content_hash,
                    _dt_to_str(created),
                    _dt_to_str(updated),
                ),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_note(self, note_id: str) -> NoteEntry | None:
        """Get a note by note_id."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM notes WHERE note_id = ?", (note_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_note_entry(row)

    def list_notes(self) -> list[NoteEntry]:
        """List all notes."""
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM notes ORDER BY note_id").fetchall()
        return [self._row_to_note_entry(row) for row in rows]

    # --- Stats ---

    def card_count(self) -> int:
        """Return total number of cards."""
        conn = self._get_connection()
        row = conn.execute("SELECT COUNT(*) as cnt FROM cards").fetchone()
        return row["cnt"]  # type: ignore[no-any-return]

    def note_count(self) -> int:
        """Return total number of notes."""
        conn = self._get_connection()
        row = conn.execute("SELECT COUNT(*) as cnt FROM notes").fetchone()
        return row["cnt"]  # type: ignore[no-any-return]

    # --- Mapping ---

    def get_mapping(self, note_id: str) -> list[CardEntry]:
        """Get all cards for a given note."""
        return self.find_cards(note_id=note_id)

    def update_mapping(self, note_id: str, cards: list[CardEntry]) -> None:
        """Replace all cards for a note."""
        conn = self._get_connection()
        conn.execute("DELETE FROM cards WHERE note_id = ?", (note_id,))
        conn.commit()
        for card in cards:
            self.add_card(card)

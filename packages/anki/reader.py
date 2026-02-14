"""Anki collection SQLite reader."""

import json
import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.anki.models import (
    AnkiCard,
    AnkiCollection,
    AnkiDeck,
    AnkiModel,
    AnkiNote,
    AnkiRevlogEntry,
    CardStats,
)


class AnkiReaderError(Exception):
    """Error reading Anki collection."""


class AnkiReader:
    """Read Anki collection from SQLite database.

    Copies the database to a temp file to avoid locking issues
    since Anki aggressively locks the database file.
    """

    def __init__(self, collection_path: str | Path) -> None:
        """Initialize reader with path to collection.anki2."""
        self.collection_path = Path(collection_path)
        if not self.collection_path.exists():
            raise AnkiReaderError(f"Collection not found: {collection_path}")

        self._temp_path: Path | None = None
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> "AnkiReader":
        """Copy database to temp and open connection."""
        # Copy to temp file to avoid locking issues
        self._temp_path = Path(tempfile.mktemp(suffix=".anki2"))
        shutil.copy2(self.collection_path, self._temp_path)

        self._conn = sqlite3.connect(str(self._temp_path))
        self._conn.row_factory = sqlite3.Row
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Close connection and remove temp file."""
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._temp_path and self._temp_path.exists():
            self._temp_path.unlink()
            self._temp_path = None

    def _ensure_connected(self) -> sqlite3.Connection:
        """Ensure we have an open connection."""
        if self._conn is None:
            raise AnkiReaderError("Reader not opened. Use 'with AnkiReader(...) as reader:'")
        return self._conn

    def read_collection(self) -> AnkiCollection:
        """Read the complete Anki collection."""
        decks = self.read_decks()
        models = self.read_models()
        notes = self.read_notes(models)
        cards = self.read_cards()
        card_stats = self.compute_card_stats()

        return AnkiCollection(
            decks=decks,
            models=models,
            notes=notes,
            cards=cards,
            card_stats=card_stats,
            collection_path=str(self.collection_path),
            extracted_at=datetime.now(UTC),
        )

    def _has_modern_schema(self) -> bool:
        """Check if the collection uses the modern Anki schema (separate tables)."""
        conn = self._ensure_connected()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='notetypes'"
        )
        return cursor.fetchone() is not None

    def read_decks(self) -> list[AnkiDeck]:
        """Read decks from collection.

        Supports both legacy (col.decks JSON) and modern (decks table) schemas.
        """
        conn = self._ensure_connected()

        if self._has_modern_schema():
            return self._read_decks_modern(conn)
        return self._read_decks_legacy(conn)

    def _read_decks_modern(self, conn: sqlite3.Connection) -> list[AnkiDeck]:
        """Read decks from the modern separate decks table."""
        cursor = conn.execute("SELECT id, name FROM decks")
        result: list[AnkiDeck] = []

        for row in cursor:
            name = row["name"]
            parent_name = None
            if "::" in name:
                parent_name = name.rsplit("::", 1)[0]

            result.append(
                AnkiDeck(
                    deck_id=row["id"],
                    name=name,
                    parent_name=parent_name,
                    config={},
                )
            )

        return result

    def _read_decks_legacy(self, conn: sqlite3.Connection) -> list[AnkiDeck]:
        """Read decks from legacy col.decks JSON blob."""
        cursor = conn.execute("SELECT decks FROM col")
        row = cursor.fetchone()

        if not row:
            return []

        decks_json = json.loads(row["decks"])
        result: list[AnkiDeck] = []

        for deck_id_str, deck_data in decks_json.items():
            deck_id = int(deck_id_str)
            name = deck_data.get("name", "")

            parent_name = None
            if "::" in name:
                parent_name = name.rsplit("::", 1)[0]

            result.append(
                AnkiDeck(
                    deck_id=deck_id,
                    name=name,
                    parent_name=parent_name,
                    config=deck_data,
                )
            )

        return result

    def read_models(self) -> list[AnkiModel]:
        """Read note types/models from collection.

        Supports both legacy (col.models JSON) and modern (notetypes + fields tables) schemas.
        """
        conn = self._ensure_connected()

        if self._has_modern_schema():
            return self._read_models_modern(conn)
        return self._read_models_legacy(conn)

    def _read_models_modern(self, conn: sqlite3.Connection) -> list[AnkiModel]:
        """Read models from the modern notetypes + fields tables."""
        # Build field lists per notetype
        fields_cursor = conn.execute("SELECT ntid, ord, name FROM fields ORDER BY ntid, ord")
        model_fields: dict[int, list[dict[str, Any]]] = {}
        for row in fields_cursor:
            ntid = row["ntid"]
            if ntid not in model_fields:
                model_fields[ntid] = []
            model_fields[ntid].append({"name": row["name"], "ord": row["ord"]})

        # Build template lists per notetype
        templates_cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='templates'"
        )
        model_templates: dict[int, list[dict[str, Any]]] = {}
        if templates_cursor.fetchone() is not None:
            tmpl_cursor = conn.execute("SELECT ntid, ord, name FROM templates ORDER BY ntid, ord")
            for row in tmpl_cursor:
                ntid = row["ntid"]
                if ntid not in model_templates:
                    model_templates[ntid] = []
                model_templates[ntid].append({"name": row["name"], "ord": row["ord"]})

        cursor = conn.execute("SELECT id, name FROM notetypes")
        result: list[AnkiModel] = []

        for row in cursor:
            model_id = row["id"]
            result.append(
                AnkiModel(
                    model_id=model_id,
                    name=row["name"],
                    fields=model_fields.get(model_id, []),
                    templates=model_templates.get(model_id, []),
                    config={},
                )
            )

        return result

    def _read_models_legacy(self, conn: sqlite3.Connection) -> list[AnkiModel]:
        """Read models from legacy col.models JSON blob."""
        cursor = conn.execute("SELECT models FROM col")
        row = cursor.fetchone()

        if not row:
            return []

        models_json = json.loads(row["models"])
        result: list[AnkiModel] = []

        for model_id_str, model_data in models_json.items():
            model_id = int(model_id_str)
            result.append(
                AnkiModel(
                    model_id=model_id,
                    name=model_data.get("name", ""),
                    fields=model_data.get("flds", []),
                    templates=model_data.get("tmpls", []),
                    config=model_data,
                )
            )

        return result

    def read_notes(self, models: list[AnkiModel] | None = None) -> list[AnkiNote]:
        """Read notes from collection.

        Args:
            models: List of models for field name lookup. If None, will read models.
        """
        conn = self._ensure_connected()

        if models is None:
            models = self.read_models()

        # Build model_id -> field names mapping
        model_fields: dict[int, list[str]] = {}
        for model in models:
            field_names = [f.get("name", f"Field{i}") for i, f in enumerate(model.fields)]
            model_fields[model.model_id] = field_names

        cursor = conn.execute(
            """
            SELECT id, mid, tags, flds, mod, usn
            FROM notes
            """
        )

        result: list[AnkiNote] = []
        for row in cursor:
            note_id = row["id"]
            model_id = row["mid"]
            tags_str = row["tags"]
            fields_str = row["flds"]
            mtime = row["mod"]
            usn = row["usn"]

            # Parse tags (space-separated)
            tags = [t.strip() for t in tags_str.split() if t.strip()]

            # Parse fields (separated by \x1f)
            fields = fields_str.split("\x1f")

            # Map to named fields
            field_names = model_fields.get(model_id, [])
            fields_json: dict[str, str] = {}
            for i, value in enumerate(fields):
                name = field_names[i] if i < len(field_names) else f"Field{i}"
                fields_json[name] = value

            result.append(
                AnkiNote(
                    note_id=note_id,
                    model_id=model_id,
                    tags=tags,
                    fields=fields,
                    fields_json=fields_json,
                    raw_fields=fields_str,
                    mtime=mtime,
                    usn=usn,
                )
            )

        return result

    def read_cards(self) -> list[AnkiCard]:
        """Read cards from collection."""
        conn = self._ensure_connected()
        cursor = conn.execute(
            """
            SELECT id, nid, did, ord, mod, usn, type, queue, due, ivl, factor, reps, lapses
            FROM cards
            """
        )

        result: list[AnkiCard] = []
        for row in cursor:
            result.append(
                AnkiCard(
                    card_id=row["id"],
                    note_id=row["nid"],
                    deck_id=row["did"],
                    ord=row["ord"],
                    mtime=row["mod"],
                    usn=row["usn"],
                    type=row["type"],
                    queue=row["queue"],
                    due=row["due"],
                    ivl=row["ivl"],
                    ease=row["factor"],
                    reps=row["reps"],
                    lapses=row["lapses"],
                )
            )

        return result

    def read_revlog(self) -> list[AnkiRevlogEntry]:
        """Read review log entries."""
        conn = self._ensure_connected()
        cursor = conn.execute(
            """
            SELECT id, cid, usn, ease, ivl, lastIvl, factor, time, type
            FROM revlog
            """
        )

        result: list[AnkiRevlogEntry] = []
        for row in cursor:
            result.append(
                AnkiRevlogEntry(
                    id=row["id"],
                    card_id=row["cid"],
                    usn=row["usn"],
                    button_chosen=row["ease"],
                    interval=row["ivl"],
                    last_interval=row["lastIvl"],
                    ease=row["factor"],
                    time_ms=row["time"],
                    type=row["type"],
                )
            )

        return result

    def compute_card_stats(self) -> list[CardStats]:
        """Compute aggregated statistics from revlog."""
        conn = self._ensure_connected()

        # Aggregate stats per card
        cursor = conn.execute(
            """
            SELECT
                cid as card_id,
                COUNT(*) as reviews,
                AVG(factor) as avg_ease,
                SUM(CASE WHEN ease = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as fail_rate,
                MAX(id) as last_review_ms,
                SUM(time) as total_time_ms
            FROM revlog
            GROUP BY cid
            """
        )

        result: list[CardStats] = []
        for row in cursor:
            last_review_at = None
            if row["last_review_ms"]:
                # revlog.id is timestamp in milliseconds
                last_review_at = datetime.fromtimestamp(
                    row["last_review_ms"] / 1000,
                    tz=UTC,
                )

            result.append(
                CardStats(
                    card_id=row["card_id"],
                    reviews=row["reviews"],
                    avg_ease=row["avg_ease"],
                    fail_rate=row["fail_rate"],
                    last_review_at=last_review_at,
                    total_time_ms=row["total_time_ms"] or 0,
                )
            )

        return result


def read_anki_collection(collection_path: str | Path) -> AnkiCollection:
    """Convenience function to read a complete Anki collection.

    Args:
        collection_path: Path to collection.anki2 file.

    Returns:
        AnkiCollection with all extracted data.
    """
    with AnkiReader(collection_path) as reader:
        return reader.read_collection()

"""Tests for card registry and mapping."""

from __future__ import annotations

from datetime import UTC, datetime

from packages.card.mapping import CardMappingEntry, NoteMapping
from packages.card.registry import (
    CardEntry,
    CardRegistry,
    NoteEntry,
    compute_content_hash,
    compute_metadata_hash,
)

# --- Helpers ---


def _make_card_entry(
    slug: str = "test-card-0-en",
    note_id: str = "note-1",
    source_path: str = "notes/test.md",
    front: str = "What is X?",
    back: str = "X is Y.",
    language: str = "en",
    tags: tuple[str, ...] = ("tag1",),
    anki_note_id: int | None = None,
) -> CardEntry:
    return CardEntry(
        slug=slug,
        note_id=note_id,
        source_path=source_path,
        front=front,
        back=back,
        content_hash=compute_content_hash(front, back),
        metadata_hash=compute_metadata_hash("Basic", list(tags)),
        language=language,
        tags=tags,
        anki_note_id=anki_note_id,
    )


def _make_note_entry(
    note_id: str = "note-1",
    source_path: str = "notes/test.md",
    title: str = "Test Note",
) -> NoteEntry:
    return NoteEntry(
        note_id=note_id,
        source_path=source_path,
        title=title,
        content_hash="abc123",
    )


# --- Schema tests ---


class TestSchema:
    def test_tables_exist_after_init(self) -> None:
        registry = CardRegistry()
        conn = registry._get_connection()
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "cards" in tables
        assert "notes" in tables
        assert "schema_version" in tables
        registry.close()

    def test_schema_version_is_correct(self) -> None:
        registry = CardRegistry()
        conn = registry._get_connection()
        row = conn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
        assert row["version"] == 2
        registry.close()

    def test_v1_to_v2_migration(self) -> None:
        """Simulate a v1 database and verify migration adds notes table."""
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        # Create v1 schema (cards + schema_version, no notes)
        conn.execute(
            """CREATE TABLE cards (
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
            )"""
        )
        conn.execute(
            """CREATE TABLE schema_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL DEFAULT 1
            )"""
        )
        conn.execute("INSERT INTO schema_version (id, version) VALUES (1, 1)")
        conn.commit()

        # Create registry pointing to same in-memory db (bypass by injecting conn)
        registry = CardRegistry()
        registry._conn = conn
        registry._ensure_schema(conn)

        # Verify notes table exists now
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        assert "notes" in tables

        # Verify version bumped
        row = conn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
        assert row["version"] == 2
        registry.close()


# --- Card CRUD ---


class TestCardCRUD:
    def test_add_and_get_card(self) -> None:
        registry = CardRegistry()
        entry = _make_card_entry()
        assert registry.add_card(entry) is True

        result = registry.get_card("test-card-0-en")
        assert result is not None
        assert result.slug == "test-card-0-en"
        assert result.note_id == "note-1"
        assert result.front == "What is X?"
        assert result.back == "X is Y."
        assert result.tags == ("tag1",)
        assert result.created_at is not None
        registry.close()

    def test_add_duplicate_slug_returns_false(self) -> None:
        registry = CardRegistry()
        entry = _make_card_entry()
        assert registry.add_card(entry) is True
        assert registry.add_card(entry) is False
        registry.close()

    def test_get_nonexistent_card_returns_none(self) -> None:
        registry = CardRegistry()
        assert registry.get_card("nonexistent") is None
        registry.close()

    def test_update_card(self) -> None:
        registry = CardRegistry()
        entry = _make_card_entry()
        registry.add_card(entry)

        updated = CardEntry(
            slug=entry.slug,
            note_id=entry.note_id,
            source_path=entry.source_path,
            front="Updated front",
            back="Updated back",
            content_hash=compute_content_hash("Updated front", "Updated back"),
            metadata_hash=entry.metadata_hash,
            language=entry.language,
            tags=entry.tags,
            anki_note_id=12345,
        )
        assert registry.update_card(updated) is True

        result = registry.get_card(entry.slug)
        assert result is not None
        assert result.front == "Updated front"
        assert result.anki_note_id == 12345
        registry.close()

    def test_update_nonexistent_returns_false(self) -> None:
        registry = CardRegistry()
        entry = _make_card_entry(slug="no-such-card-0-en")
        assert registry.update_card(entry) is False
        registry.close()

    def test_delete_card(self) -> None:
        registry = CardRegistry()
        entry = _make_card_entry()
        registry.add_card(entry)
        assert registry.delete_card(entry.slug) is True
        assert registry.get_card(entry.slug) is None
        registry.close()

    def test_delete_nonexistent_returns_false(self) -> None:
        registry = CardRegistry()
        assert registry.delete_card("nonexistent") is False
        registry.close()

    def test_find_cards_by_note_id(self) -> None:
        registry = CardRegistry()
        registry.add_card(_make_card_entry(slug="card-a-0-en", note_id="note-1"))
        registry.add_card(_make_card_entry(slug="card-b-0-en", note_id="note-1"))
        registry.add_card(_make_card_entry(slug="card-c-0-en", note_id="note-2"))

        results = registry.find_cards(note_id="note-1")
        assert len(results) == 2
        assert {r.slug for r in results} == {"card-a-0-en", "card-b-0-en"}
        registry.close()

    def test_find_cards_by_source_path(self) -> None:
        registry = CardRegistry()
        registry.add_card(_make_card_entry(slug="card-a-0-en", source_path="notes/a.md"))
        registry.add_card(_make_card_entry(slug="card-b-0-en", source_path="notes/b.md"))

        results = registry.find_cards(source_path="notes/a.md")
        assert len(results) == 1
        assert results[0].slug == "card-a-0-en"
        registry.close()

    def test_find_cards_by_content_hash(self) -> None:
        registry = CardRegistry()
        entry = _make_card_entry()
        registry.add_card(entry)

        results = registry.find_cards(content_hash=entry.content_hash)
        assert len(results) == 1
        assert results[0].slug == entry.slug
        registry.close()

    def test_find_cards_no_filters(self) -> None:
        registry = CardRegistry()
        registry.add_card(_make_card_entry(slug="card-a-0-en"))
        registry.add_card(_make_card_entry(slug="card-b-0-en"))

        results = registry.find_cards()
        assert len(results) == 2
        registry.close()


# --- Note CRUD ---


class TestNoteCRUD:
    def test_add_and_get_note(self) -> None:
        registry = CardRegistry()
        entry = _make_note_entry()
        assert registry.add_note(entry) is True

        result = registry.get_note("note-1")
        assert result is not None
        assert result.note_id == "note-1"
        assert result.source_path == "notes/test.md"
        assert result.title == "Test Note"
        assert result.content_hash == "abc123"
        registry.close()

    def test_add_duplicate_note_returns_false(self) -> None:
        registry = CardRegistry()
        entry = _make_note_entry()
        assert registry.add_note(entry) is True
        assert registry.add_note(entry) is False
        registry.close()

    def test_get_nonexistent_note_returns_none(self) -> None:
        registry = CardRegistry()
        assert registry.get_note("nonexistent") is None
        registry.close()

    def test_list_notes(self) -> None:
        registry = CardRegistry()
        registry.add_note(_make_note_entry(note_id="note-1", source_path="a.md"))
        registry.add_note(_make_note_entry(note_id="note-2", source_path="b.md"))

        results = registry.list_notes()
        assert len(results) == 2
        assert results[0].note_id == "note-1"
        assert results[1].note_id == "note-2"
        registry.close()


# --- Stats ---


class TestStats:
    def test_card_count(self) -> None:
        registry = CardRegistry()
        assert registry.card_count() == 0
        registry.add_card(_make_card_entry(slug="card-a-0-en"))
        registry.add_card(_make_card_entry(slug="card-b-0-en"))
        assert registry.card_count() == 2
        registry.close()

    def test_note_count(self) -> None:
        registry = CardRegistry()
        assert registry.note_count() == 0
        registry.add_note(_make_note_entry(note_id="n1", source_path="a.md"))
        assert registry.note_count() == 1
        registry.close()


# --- Mapping ---


class TestMapping:
    def test_get_mapping(self) -> None:
        registry = CardRegistry()
        registry.add_card(_make_card_entry(slug="card-a-0-en", note_id="note-1"))
        registry.add_card(_make_card_entry(slug="card-b-0-en", note_id="note-1"))
        registry.add_card(_make_card_entry(slug="card-c-0-en", note_id="note-2"))

        mapping = registry.get_mapping("note-1")
        assert len(mapping) == 2
        assert {c.slug for c in mapping} == {"card-a-0-en", "card-b-0-en"}
        registry.close()

    def test_update_mapping_replaces_cards(self) -> None:
        registry = CardRegistry()
        registry.add_card(_make_card_entry(slug="old-card-0-en", note_id="note-1"))

        new_cards = [
            _make_card_entry(slug="new-card-a-0-en", note_id="note-1"),
            _make_card_entry(slug="new-card-b-0-en", note_id="note-1"),
        ]
        registry.update_mapping("note-1", new_cards)

        mapping = registry.get_mapping("note-1")
        assert len(mapping) == 2
        assert {c.slug for c in mapping} == {"new-card-a-0-en", "new-card-b-0-en"}
        assert registry.get_card("old-card-0-en") is None
        registry.close()


# --- Hash helpers ---


class TestHashHelpers:
    def test_compute_content_hash_deterministic(self) -> None:
        h1 = compute_content_hash("front", "back")
        h2 = compute_content_hash("front", "back")
        assert h1 == h2
        assert len(h1) == 12

    def test_compute_content_hash_different_for_different_content(self) -> None:
        h1 = compute_content_hash("front1", "back1")
        h2 = compute_content_hash("front2", "back2")
        assert h1 != h2

    def test_compute_metadata_hash_deterministic(self) -> None:
        h1 = compute_metadata_hash("Basic", ["tag1", "tag2"])
        h2 = compute_metadata_hash("Basic", ["tag2", "tag1"])
        assert h1 == h2
        assert len(h1) == 6

    def test_compute_metadata_hash_different_for_different_types(self) -> None:
        h1 = compute_metadata_hash("Basic", ["tag1"])
        h2 = compute_metadata_hash("Cloze", ["tag1"])
        assert h1 != h2


# --- Mapping dataclasses ---


class TestCardMappingEntry:
    def test_is_synced_true(self) -> None:
        entry = CardMappingEntry(slug="test-0-en", language="en", anki_note_id=123)
        assert entry.is_synced is True

    def test_is_synced_false(self) -> None:
        entry = CardMappingEntry(slug="test-0-en", language="en")
        assert entry.is_synced is False

    def test_from_card_entry(self) -> None:
        card = _make_card_entry(anki_note_id=42)
        mapping = CardMappingEntry.from_card_entry(card)
        assert mapping.slug == card.slug
        assert mapping.language == card.language
        assert mapping.anki_note_id == 42
        assert mapping.content_hash == card.content_hash


class TestNoteMapping:
    def test_card_counts(self) -> None:
        cards = (
            CardMappingEntry(slug="a-0-en", language="en", anki_note_id=1),
            CardMappingEntry(slug="b-0-en", language="en"),
            CardMappingEntry(slug="c-0-en", language="en", anki_note_id=2),
        )
        mapping = NoteMapping(
            note_path="notes/test.md",
            note_id="note-1",
            note_title="Test",
            cards=cards,
        )
        assert mapping.card_count == 3
        assert mapping.synced_count == 2
        assert mapping.unsynced_count == 1

    def test_empty_mapping(self) -> None:
        mapping = NoteMapping(
            note_path="notes/test.md",
            note_id="note-1",
            note_title="Test",
        )
        assert mapping.card_count == 0
        assert mapping.synced_count == 0
        assert mapping.unsynced_count == 0

    def test_orphan_flag(self) -> None:
        mapping = NoteMapping(
            note_path="notes/test.md",
            note_id="note-1",
            note_title="Test",
            is_orphan=True,
        )
        assert mapping.is_orphan is True

    def test_last_sync(self) -> None:
        ts = datetime(2026, 1, 1, tzinfo=UTC)
        mapping = NoteMapping(
            note_path="notes/test.md",
            note_id="note-1",
            note_title="Test",
            last_sync=ts,
        )
        assert mapping.last_sync == ts

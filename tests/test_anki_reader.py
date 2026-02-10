"""Tests for Anki SQLite reader."""

from pathlib import Path

import pytest

from packages.anki.reader import AnkiReader, AnkiReaderError, read_anki_collection


def test_reader_requires_existing_file(temp_dir: Path) -> None:
    """Test that reader raises error for non-existent file."""
    with pytest.raises(AnkiReaderError, match="Collection not found"):
        AnkiReader(temp_dir / "nonexistent.anki2")


def test_reader_context_manager(sample_collection: Path) -> None:
    """Test reader works as context manager."""
    with AnkiReader(sample_collection) as reader:
        assert reader._conn is not None

    # Connection should be closed after context
    assert reader._conn is None


def test_read_decks(sample_collection: Path) -> None:
    """Test reading decks from collection."""
    with AnkiReader(sample_collection) as reader:
        decks = reader.read_decks()

    assert len(decks) == 2

    # Check Default deck
    default_deck = next((d for d in decks if d.name == "Default"), None)
    assert default_deck is not None
    assert default_deck.deck_id == 1
    assert default_deck.parent_name is None

    # Check nested deck
    python_deck = next((d for d in decks if d.name == "Programming::Python"), None)
    assert python_deck is not None
    assert python_deck.deck_id == 1234567890
    assert python_deck.parent_name == "Programming"


def test_read_models(sample_collection: Path) -> None:
    """Test reading note models from collection."""
    with AnkiReader(sample_collection) as reader:
        models = reader.read_models()

    assert len(models) == 1

    model = models[0]
    assert model.name == "Basic"
    assert len(model.fields) == 2
    assert model.fields[0]["name"] == "Front"
    assert model.fields[1]["name"] == "Back"


def test_read_notes(sample_collection: Path) -> None:
    """Test reading notes from collection."""
    with AnkiReader(sample_collection) as reader:
        notes = reader.read_notes()

    assert len(notes) == 3

    # Check first note
    note1 = next((n for n in notes if n.note_id == 1000000001), None)
    assert note1 is not None
    assert note1.model_id == 1234567891
    assert "python" in note1.tags
    assert "programming" in note1.tags
    assert len(note1.fields) == 2
    assert "list" in note1.fields[0]
    assert note1.fields_json["Front"] == "What is a <b>list</b> in Python?"
    assert note1.fields_json["Back"] == "An ordered, mutable collection of items."


def test_read_cards(sample_collection: Path) -> None:
    """Test reading cards from collection."""
    with AnkiReader(sample_collection) as reader:
        cards = reader.read_cards()

    assert len(cards) == 3

    # Check mature card
    card1 = next((c for c in cards if c.card_id == 2000000001), None)
    assert card1 is not None
    assert card1.note_id == 1000000001
    assert card1.deck_id == 1234567890
    assert card1.ivl == 21  # 21 days = mature
    assert card1.ease == 2500  # 250%
    assert card1.reps == 10
    assert card1.lapses == 2


def test_compute_card_stats(sample_collection: Path) -> None:
    """Test computing card statistics from revlog."""
    with AnkiReader(sample_collection) as reader:
        stats = reader.compute_card_stats()

    assert len(stats) == 2  # Only 2 cards have reviews

    # Card 1 has 3 reviews, 1 fail
    card1_stats = next((s for s in stats if s.card_id == 2000000001), None)
    assert card1_stats is not None
    assert card1_stats.reviews == 3
    assert card1_stats.fail_rate == pytest.approx(1 / 3, rel=0.01)
    assert card1_stats.total_time_ms == 16000  # 5000 + 3000 + 8000

    # Card 2 has 1 review, no fails
    card2_stats = next((s for s in stats if s.card_id == 2000000002), None)
    assert card2_stats is not None
    assert card2_stats.reviews == 1
    assert card2_stats.fail_rate == 0.0


def test_read_collection_convenience(sample_collection: Path) -> None:
    """Test convenience function for reading full collection."""
    collection = read_anki_collection(sample_collection)

    assert len(collection.decks) == 2
    assert len(collection.models) == 1
    assert len(collection.notes) == 3
    assert len(collection.cards) == 3
    assert len(collection.card_stats) == 2
    assert collection.collection_path == str(sample_collection)

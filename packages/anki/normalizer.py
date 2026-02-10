"""HTML to text normalization for Anki fields."""

import html
import re
from typing import Any

from packages.anki.models import AnkiNote

# Regex patterns
CLOZE_PATTERN = re.compile(r"\{\{c\d+::([^}]+?)(?:::[^}]+)?\}\}")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
CODE_BLOCK_PATTERN = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL | re.IGNORECASE)
CODE_INLINE_PATTERN = re.compile(r"<code[^>]*>(.*?)</code>", re.DOTALL | re.IGNORECASE)

# Common Anki field names for front/back detection
FRONT_FIELDS = {"front", "question", "expression", "word", "term", "prompt"}
BACK_FIELDS = {"back", "answer", "meaning", "definition", "response", "reading"}
EXTRA_FIELDS = {"extra", "notes", "hint", "example", "examples", "context"}


def strip_html(text: str, preserve_code: bool = True) -> str:
    """Strip HTML tags from text.

    Args:
        text: HTML text to strip.
        preserve_code: If True, preserve content of <code> and <pre> blocks.

    Returns:
        Plain text with HTML removed.
    """
    if not text:
        return ""

    result = text

    if preserve_code:
        # Replace code blocks with placeholders
        code_blocks: list[str] = []

        def save_code_block(match: re.Match[str]) -> str:
            code_blocks.append(match.group(1))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

        result = CODE_BLOCK_PATTERN.sub(save_code_block, result)
        result = CODE_INLINE_PATTERN.sub(save_code_block, result)

    # Handle cloze deletions - extract the answer
    result = CLOZE_PATTERN.sub(r"\1", result)

    # Replace common HTML entities
    result = result.replace("<br>", "\n")
    result = result.replace("<br/>", "\n")
    result = result.replace("<br />", "\n")
    result = result.replace("&nbsp;", " ")
    result = result.replace("<p>", "\n")
    result = result.replace("</p>", "\n")
    result = result.replace("<div>", "\n")
    result = result.replace("</div>", "\n")
    result = result.replace("<li>", "\n- ")
    result = result.replace("</li>", "")

    # Strip remaining HTML tags
    result = HTML_TAG_PATTERN.sub("", result)

    # Decode HTML entities
    result = html.unescape(result)

    if preserve_code:
        # Restore code blocks
        for i, code in enumerate(code_blocks):
            # Decode entities in code too
            decoded_code = html.unescape(code)
            result = result.replace(f"__CODE_BLOCK_{i}__", f"`{decoded_code}`")

    # Normalize whitespace
    result = WHITESPACE_PATTERN.sub(" ", result)

    return result.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple spaces/tabs with single space
    result = WHITESPACE_PATTERN.sub(" ", text)
    # But preserve single newlines
    lines = result.split("\n")
    lines = [line.strip() for line in lines]
    return "\n".join(line for line in lines if line)


def classify_field(name: str) -> str:
    """Classify field as front, back, extra, or other."""
    name_lower = name.lower()
    if name_lower in FRONT_FIELDS:
        return "front"
    if name_lower in BACK_FIELDS:
        return "back"
    if name_lower in EXTRA_FIELDS:
        return "extra"
    return "other"


def normalize_note(
    note: AnkiNote,
    deck_names: list[str] | None = None,
) -> str:
    """Normalize a note's fields to searchable text.

    Produces a deterministic template:
        Front: ...
        Back: ...
        Extra: ...
        Tags: tag1, tag2
        Decks: Deck1, Deck2

    Args:
        note: The note to normalize.
        deck_names: Optional list of deck names for this note.

    Returns:
        Normalized text suitable for search indexing.
    """
    parts: list[str] = []

    # Classify and organize fields
    front_parts: list[str] = []
    back_parts: list[str] = []
    extra_parts: list[str] = []
    other_parts: list[str] = []

    for field_name, field_value in note.fields_json.items():
        stripped = strip_html(field_value)
        if not stripped:
            continue

        classification = classify_field(field_name)
        if classification == "front":
            front_parts.append(stripped)
        elif classification == "back":
            back_parts.append(stripped)
        elif classification == "extra":
            extra_parts.append(stripped)
        else:
            other_parts.append(stripped)

    # If no explicit front/back, use first two fields
    if not front_parts and not back_parts and note.fields:
        if len(note.fields) >= 1:
            front_parts = [strip_html(note.fields[0])]
        if len(note.fields) >= 2:
            back_parts = [strip_html(note.fields[1])]
        if len(note.fields) >= 3:
            extra_parts = [strip_html(f) for f in note.fields[2:] if strip_html(f)]

    # Build normalized text
    if front_parts:
        parts.append(f"Front: {' '.join(front_parts)}")
    if back_parts:
        parts.append(f"Back: {' '.join(back_parts)}")
    if extra_parts:
        parts.append(f"Extra: {' '.join(extra_parts)}")
    if other_parts:
        # Include other fields without label
        parts.extend(other_parts)

    # Add tags
    if note.tags:
        parts.append(f"Tags: {', '.join(note.tags)}")

    # Add deck names
    if deck_names:
        parts.append(f"Decks: {', '.join(deck_names)}")

    result = "\n".join(parts)
    return normalize_whitespace(result)


def normalize_notes(
    notes: list[AnkiNote],
    deck_map: dict[int, str] | None = None,
    card_deck_map: dict[int, list[int]] | None = None,
) -> list[AnkiNote]:
    """Normalize all notes in a collection.

    Args:
        notes: List of notes to normalize.
        deck_map: Mapping of deck_id -> deck_name.
        card_deck_map: Mapping of note_id -> list of deck_ids for that note's cards.

    Returns:
        Notes with normalized_text populated.
    """
    for note in notes:
        # Get deck names for this note
        deck_names: list[str] = []
        if deck_map and card_deck_map:
            deck_ids = card_deck_map.get(note.note_id, [])
            deck_names = [deck_map[did] for did in deck_ids if did in deck_map]

        note.normalized_text = normalize_note(note, deck_names)

    return notes


def build_deck_map(decks: list[Any]) -> dict[int, str]:
    """Build mapping of deck_id -> deck_name."""
    return {d.deck_id: d.name for d in decks}


def build_card_deck_map(cards: list[Any]) -> dict[int, list[int]]:
    """Build mapping of note_id -> list of deck_ids."""
    result: dict[int, list[int]] = {}
    for card in cards:
        if card.note_id not in result:
            result[card.note_id] = []
        if card.deck_id not in result[card.note_id]:
            result[card.note_id].append(card.deck_id)
    return result

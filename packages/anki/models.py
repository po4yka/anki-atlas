"""Anki data models."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnkiDeck(BaseModel):
    """Anki deck from collection."""

    deck_id: int
    name: str
    parent_name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class AnkiModel(BaseModel):
    """Anki note type/model from collection."""

    model_id: int
    name: str
    fields: list[dict[str, Any]]
    templates: list[dict[str, Any]]
    config: dict[str, Any] = Field(default_factory=dict)


class AnkiNote(BaseModel):
    """Anki note from collection."""

    note_id: int
    model_id: int
    tags: list[str] = Field(default_factory=list)
    fields: list[str]  # Raw field values from Anki (separated by \x1f)
    fields_json: dict[str, str] = Field(default_factory=dict)  # Named fields
    raw_fields: str | None = None
    normalized_text: str = ""
    mtime: int  # Modification timestamp (seconds since epoch)
    usn: int  # Update sequence number


class AnkiCard(BaseModel):
    """Anki card from collection."""

    card_id: int
    note_id: int
    deck_id: int
    ord: int = 0  # Card ordinal within note
    due: int | None = None
    ivl: int = 0  # Interval in days
    ease: int = 0  # Ease factor (permille, e.g., 2500 = 250%)
    lapses: int = 0
    reps: int = 0
    queue: int = 0  # -3=user buried, -2=sched buried, -1=suspended, 0=new, 1=learning, 2=review, 3=in learning
    type: int = 0  # 0=new, 1=learning, 2=review, 3=relearning
    mtime: int
    usn: int


class AnkiRevlogEntry(BaseModel):
    """Anki review log entry."""

    id: int  # Timestamp in milliseconds
    card_id: int
    usn: int
    button_chosen: int  # 1=again, 2=hard, 3=good, 4=easy
    interval: int  # New interval
    last_interval: int  # Previous interval
    ease: int  # New ease factor
    time_ms: int  # Time spent on review (ms)
    type: int  # 0=learn, 1=review, 2=relearn, 3=filtered


class CardStats(BaseModel):
    """Aggregated card statistics from revlog."""

    card_id: int
    reviews: int = 0
    avg_ease: float | None = None
    fail_rate: float | None = None
    last_review_at: datetime | None = None
    total_time_ms: int = 0


class AnkiCollection(BaseModel):
    """Complete extracted Anki collection."""

    decks: list[AnkiDeck]
    models: list[AnkiModel]
    notes: list[AnkiNote]
    cards: list[AnkiCard]
    card_stats: list[CardStats] = Field(default_factory=list)

    # Metadata
    collection_path: str | None = None
    extracted_at: datetime = Field(default_factory=datetime.now)
    schema_version: int = 11  # Anki schema version

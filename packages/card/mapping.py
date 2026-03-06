"""Mapping data classes for card-to-note relationships."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from packages.card.registry import CardEntry


@dataclass(frozen=True, slots=True)
class CardMappingEntry:
    """Lightweight mapping entry for a card."""

    slug: str
    language: str
    anki_note_id: int | None = None
    synced_at: datetime | None = None
    content_hash: str = ""

    @property
    def is_synced(self) -> bool:
        """Check if the card has been synced to Anki."""
        return self.anki_note_id is not None

    @classmethod
    def from_card_entry(cls, entry: CardEntry) -> CardMappingEntry:
        """Create a mapping entry from a CardEntry."""
        return cls(
            slug=entry.slug,
            language=entry.language,
            anki_note_id=entry.anki_note_id,
            synced_at=entry.synced_at,
            content_hash=entry.content_hash,
        )


@dataclass(frozen=True, slots=True)
class NoteMapping:
    """Mapping of a note to its cards."""

    note_path: str
    note_id: str
    note_title: str
    cards: tuple[CardMappingEntry, ...] = ()
    last_sync: datetime | None = None
    is_orphan: bool = False

    @property
    def card_count(self) -> int:
        """Total number of cards."""
        return len(self.cards)

    @property
    def synced_count(self) -> int:
        """Number of synced cards."""
        return sum(1 for c in self.cards if c.is_synced)

    @property
    def unsynced_count(self) -> int:
        """Number of unsynced cards."""
        return self.card_count - self.synced_count

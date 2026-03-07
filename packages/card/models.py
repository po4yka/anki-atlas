"""Card domain entities for Anki Atlas."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from packages.common.exceptions import CardValidationError
from packages.common.types import Language

VALID_NOTE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "APF::Simple",
        "APF::Cloze",
        "Basic",
        "Cloze",
    }
)

_VALID_LANG_VALUES: Final[frozenset[str]] = frozenset(lang.value for lang in Language)


@dataclass(frozen=True, slots=True)
class CardManifest:
    """Value object linking a card to its source note."""

    slug: str
    slug_base: str
    lang: str
    source_path: str
    source_anchor: str
    note_id: str
    note_title: str
    card_index: int
    guid: str | None = None
    hash6: str | None = None
    obsidian_uri: str | None = None
    difficulty: float | None = None
    cognitive_load: str | None = None

    def __post_init__(self) -> None:
        """Validate manifest invariants."""
        errors: list[str] = []

        if not self.slug:
            errors.append("slug cannot be empty")

        if not self.slug_base:
            errors.append("slug_base cannot be empty")

        if not self.lang:
            errors.append("lang cannot be empty")
        elif len(self.lang) != 2:
            errors.append(f"lang must be 2 characters, got '{self.lang}'")
        elif self.lang not in _VALID_LANG_VALUES:
            errors.append(
                f"lang '{self.lang}' not in valid languages: {sorted(_VALID_LANG_VALUES)}"
            )

        if not self.source_path:
            errors.append("source_path cannot be empty")

        if not self.source_anchor:
            errors.append("source_anchor cannot be empty")

        if not self.note_id:
            errors.append("note_id cannot be empty")

        if not self.note_title:
            errors.append("note_title cannot be empty")

        if self.card_index < 0:
            errors.append(f"card_index must be >= 0, got {self.card_index}")

        if self.hash6 is not None:
            if len(self.hash6) != 6:
                errors.append(f"hash6 must be exactly 6 characters, got {len(self.hash6)}")
            elif not all(c in "0123456789abcdef" for c in self.hash6.lower()):
                errors.append("hash6 must be a valid hexadecimal string")

        if self.difficulty is not None and not 0.0 <= self.difficulty <= 1.0:
            errors.append(f"difficulty must be between 0.0 and 1.0, got {self.difficulty}")

        if self.cognitive_load is not None:
            valid_loads = ("basic", "intermediate", "advanced")
            if self.cognitive_load not in valid_loads:
                errors.append(
                    f"cognitive_load must be one of {valid_loads}, got '{self.cognitive_load}'"
                )

        if errors:
            raise CardValidationError(f"CardManifest validation failed: {'; '.join(errors)}")

    @property
    def anchor_url(self) -> str:
        """Generate Obsidian wikilink: [[folder/note#anchor]]."""
        path = self.source_path
        if "/" in path:
            parts = path.rstrip("/").split("/")
            relative = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            path = relative
        return f"[[{path}#{self.source_anchor}]]"

    @property
    def is_linked_to_note(self) -> bool:
        """Check if manifest links to a valid note."""
        return bool(self.note_id and self.source_path)

    def with_guid(self, guid: str) -> CardManifest:
        """Create new manifest with Anki GUID."""
        return CardManifest(
            slug=self.slug,
            slug_base=self.slug_base,
            lang=self.lang,
            source_path=self.source_path,
            source_anchor=self.source_anchor,
            note_id=self.note_id,
            note_title=self.note_title,
            card_index=self.card_index,
            guid=guid,
            hash6=self.hash6,
            obsidian_uri=self.obsidian_uri,
            difficulty=self.difficulty,
            cognitive_load=self.cognitive_load,
        )

    def with_hash(self, hash6: str) -> CardManifest:
        """Create new manifest with content hash."""
        return CardManifest(
            slug=self.slug,
            slug_base=self.slug_base,
            lang=self.lang,
            source_path=self.source_path,
            source_anchor=self.source_anchor,
            note_id=self.note_id,
            note_title=self.note_title,
            card_index=self.card_index,
            guid=self.guid,
            hash6=hash6,
            obsidian_uri=self.obsidian_uri,
            difficulty=self.difficulty,
            cognitive_load=self.cognitive_load,
        )

    def with_obsidian_uri(self, obsidian_uri: str) -> CardManifest:
        """Create new manifest with Obsidian URI."""
        return CardManifest(
            slug=self.slug,
            slug_base=self.slug_base,
            lang=self.lang,
            source_path=self.source_path,
            source_anchor=self.source_anchor,
            note_id=self.note_id,
            note_title=self.note_title,
            card_index=self.card_index,
            guid=self.guid,
            hash6=self.hash6,
            obsidian_uri=obsidian_uri,
            difficulty=self.difficulty,
            cognitive_load=self.cognitive_load,
        )

    def with_fsrs_metadata(self, difficulty: float, cognitive_load: str) -> CardManifest:
        """Create new manifest with FSRS metadata."""
        return CardManifest(
            slug=self.slug,
            slug_base=self.slug_base,
            lang=self.lang,
            source_path=self.source_path,
            source_anchor=self.source_anchor,
            note_id=self.note_id,
            note_title=self.note_title,
            card_index=self.card_index,
            guid=self.guid,
            hash6=self.hash6,
            obsidian_uri=self.obsidian_uri,
            difficulty=difficulty,
            cognitive_load=cognitive_load,
        )


@dataclass(frozen=True, slots=True)
class Card:
    """Domain entity representing an Anki flashcard."""

    slug: str
    language: str
    apf_html: str
    manifest: CardManifest
    note_type: str
    tags: list[str] = field(default_factory=list)
    anki_guid: str | None = None

    def __post_init__(self) -> None:
        """Validate entity invariants."""
        errors: list[str] = []

        if not self.slug:
            errors.append("slug cannot be empty")
        elif len(self.slug) < 3:
            errors.append(f"slug must be at least 3 characters, got '{self.slug}'")

        if not self.language:
            errors.append("language cannot be empty")
        elif len(self.language) != 2:
            errors.append(f"language must be 2 characters, got '{self.language}'")
        elif self.language not in _VALID_LANG_VALUES:
            errors.append(
                f"language '{self.language}' not in valid languages: {sorted(_VALID_LANG_VALUES)}"
            )

        if not self.apf_html:
            errors.append("apf_html cannot be empty")
        elif len(self.apf_html.strip()) < 10:
            errors.append("apf_html content is too short (min 10 characters)")

        if not self.note_type:
            errors.append("note_type cannot be empty")
        elif self.note_type not in VALID_NOTE_TYPES:
            errors.append(
                f"note_type '{self.note_type}' not in valid types: {sorted(VALID_NOTE_TYPES)}"
            )

        if self.manifest.lang != self.language:
            errors.append(
                f"manifest.lang '{self.manifest.lang}' does not match "
                f"card language '{self.language}'"
            )

        if self.manifest.slug != self.slug:
            errors.append(
                f"manifest.slug '{self.manifest.slug}' does not match card slug '{self.slug}'"
            )

        for i, tag in enumerate(self.tags):
            if not tag or not tag.strip():
                errors.append(f"tag at index {i} cannot be empty")

        if errors:
            raise CardValidationError(f"Card validation failed: {'; '.join(errors)}")

    @property
    def is_new(self) -> bool:
        """Check if card is new (not yet in Anki)."""
        return self.anki_guid is None

    @property
    def content_hash(self) -> str:
        """Calculate 6-character content hash for change detection."""
        content = f"{self.apf_html}|{self.note_type}|{','.join(sorted(self.tags))}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:6]

    def with_guid(self, guid: str) -> Card:
        """Create new Card with Anki GUID (immutable pattern)."""
        if not guid or not guid.strip():
            raise CardValidationError("guid cannot be empty")
        new_manifest = self.manifest.with_guid(guid)
        return Card(
            slug=self.slug,
            language=self.language,
            apf_html=self.apf_html,
            manifest=new_manifest,
            note_type=self.note_type,
            tags=list(self.tags),
            anki_guid=guid,
        )

    def update_content(self, new_apf_html: str) -> Card:
        """Create new Card with updated content and recalculated hash."""
        if not new_apf_html or not new_apf_html.strip():
            raise CardValidationError("new_apf_html cannot be empty")
        new_card = Card(
            slug=self.slug,
            language=self.language,
            apf_html=new_apf_html,
            manifest=self.manifest,
            note_type=self.note_type,
            tags=list(self.tags),
            anki_guid=self.anki_guid,
        )
        new_manifest = self.manifest.with_hash(new_card.content_hash)
        return Card(
            slug=self.slug,
            language=self.language,
            apf_html=new_apf_html,
            manifest=new_manifest,
            note_type=self.note_type,
            tags=list(self.tags),
            anki_guid=self.anki_guid,
        )

    def with_tags(self, tags: list[str]) -> Card:
        """Create new Card with updated tags."""
        return Card(
            slug=self.slug,
            language=self.language,
            apf_html=self.apf_html,
            manifest=self.manifest,
            note_type=self.note_type,
            tags=list(tags),
            anki_guid=self.anki_guid,
        )


class SyncActionType(Enum):
    """Possible sync action types."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class SyncAction:
    """Domain entity representing a sync action for a card."""

    action_type: SyncActionType
    card: Card
    anki_guid: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate sync action invariants."""
        errors: list[str] = []

        if not isinstance(self.action_type, SyncActionType):
            errors.append(f"action_type must be SyncActionType, got {type(self.action_type)}")

        if not isinstance(self.card, Card):
            errors.append(f"card must be Card instance, got {type(self.card)}")

        if (
            self.action_type in (SyncActionType.UPDATE, SyncActionType.DELETE)
            and not self.anki_guid
        ):
            errors.append(f"anki_guid is required for {self.action_type.value} actions")

        if errors:
            raise CardValidationError(f"SyncAction validation failed: {'; '.join(errors)}")

    @property
    def is_destructive(self) -> bool:
        """Check if this action modifies or deletes data in Anki."""
        return self.action_type in (SyncActionType.UPDATE, SyncActionType.DELETE)

    @property
    def requires_confirmation(self) -> bool:
        """Check if this action requires user confirmation."""
        return self.action_type == SyncActionType.DELETE

    def describe(self) -> str:
        """Generate a human-readable description of this action."""
        base = f"{self.action_type.value.upper()}: {self.card.slug}"
        if self.reason:
            base += f" ({self.reason})"
        return base

"""Obsidian sync workflow: discover -> generate -> validate -> sync."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from packages.common.logging import get_logger
from packages.obsidian.parser import ParsedNote, discover_notes, parse_note

if TYPE_CHECKING:
    from pathlib import Path

    from packages.anki.sync.engine import SyncEngine
    from packages.generator.agents.models import GeneratedCard
    from packages.validation.pipeline import ValidationPipeline

logger = get_logger(module=__name__)

ProgressCallback = Callable[[str, int, int], None]


class CardGeneratorProtocol(Protocol):
    """Protocol for card generation from a parsed note."""

    def generate(self, note: ParsedNote) -> list[GeneratedCard]: ...


@dataclass(frozen=True, slots=True)
class SyncResult:
    """Workflow-level sync result with counts and errors."""

    created: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    errors: tuple[str, ...] = ()

    def merge(self, other: SyncResult) -> SyncResult:
        """Combine two results."""
        return SyncResult(
            created=self.created + other.created,
            updated=self.updated + other.updated,
            skipped=self.skipped + other.skipped,
            failed=self.failed + other.failed,
            errors=self.errors + other.errors,
        )


@dataclass(slots=True)
class NoteResult:
    """Result of processing a single note."""

    note_path: Path
    cards: list[GeneratedCard] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ObsidianSyncWorkflow:
    """Orchestrates: discover notes -> generate cards -> validate -> sync to Anki.

    Thin orchestrator. Each step is independently callable.
    """

    def __init__(
        self,
        generator: CardGeneratorProtocol,
        validator: ValidationPipeline | None = None,
        sync_engine: SyncEngine | None = None,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._generator = generator
        self._validator = validator
        self._sync_engine = sync_engine
        self._on_progress = on_progress

    def _report(self, phase: str, current: int, total: int) -> None:
        if self._on_progress is not None:
            self._on_progress(phase, current, total)

    def scan_vault(
        self,
        vault_path: Path,
        *,
        source_dirs: Sequence[str] | None = None,
    ) -> list[ParsedNote]:
        """Discover and parse all notes in vault."""
        if source_dirs is not None:
            all_notes: list[Path] = []
            for d in source_dirs:
                dir_path = vault_path / d
                if dir_path.is_dir():
                    all_notes.extend(discover_notes(dir_path))
        else:
            all_notes = discover_notes(vault_path)

        parsed: list[ParsedNote] = []
        total = len(all_notes)
        for i, note_path in enumerate(all_notes):
            self._report("scan", i + 1, total)
            try:
                parsed.append(parse_note(note_path, vault_root=vault_path))
            except Exception as exc:
                logger.warning("scan.parse_failed", path=str(note_path), error=str(exc))
        logger.info("scan.complete", notes=len(parsed), vault=str(vault_path))
        return parsed

    def process_note(self, note: ParsedNote) -> NoteResult:
        """Generate cards from a parsed note, optionally validating each."""
        result = NoteResult(note_path=note.path)
        try:
            cards = self._generator.generate(note)
        except Exception as exc:
            logger.warning("process.generate_failed", path=str(note.path), error=str(exc))
            result.errors.append(f"Generation failed for {note.path}: {exc}")
            return result

        if self._validator is None:
            result.cards = cards
            return result

        for card in cards:
            vr = self._validator.run(front=card.apf_html, back="", tags=())
            if vr.is_valid:
                result.cards.append(card)
            else:
                msgs = "; ".join(i.message for i in vr.errors())
                result.errors.append(f"Validation failed for card {card.slug}: {msgs}")
                logger.debug("process.validation_failed", slug=card.slug, errors=msgs)

        return result

    def run(
        self,
        vault_path: Path,
        *,
        source_dirs: Sequence[str] | None = None,
    ) -> SyncResult:
        """Full pipeline: scan -> process all notes -> aggregate results."""
        notes = self.scan_vault(vault_path, source_dirs=source_dirs)
        total = len(notes)

        all_cards: list[GeneratedCard] = []
        all_errors: list[str] = []
        failed = 0
        skipped = 0

        for i, note in enumerate(notes):
            self._report("process", i + 1, total)
            nr = self.process_note(note)
            all_cards.extend(nr.cards)
            all_errors.extend(nr.errors)
            if nr.errors and not nr.cards:
                failed += 1
            elif nr.errors:
                skipped += len(nr.errors)

        logger.info(
            "run.complete",
            total_notes=total,
            total_cards=len(all_cards),
            failed=failed,
        )

        return SyncResult(
            created=len(all_cards),
            skipped=skipped,
            failed=failed,
            errors=tuple(all_errors),
        )

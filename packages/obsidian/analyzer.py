"""Vault structure analysis using pathlib and regex (no obsidiantools dependency)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from packages.common.logging import get_logger
from packages.obsidian.parser import discover_notes

logger = get_logger(module=__name__)

# Matches [[target]] and [[target|alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


@dataclass(frozen=True, slots=True)
class VaultStats:
    """Statistics about an Obsidian vault."""

    total_notes: int
    total_dirs: int
    notes_with_frontmatter: int
    wikilinks_count: int
    orphaned_notes: tuple[str, ...]
    broken_links: tuple[tuple[str, str], ...]


class VaultAnalyzer:
    """Analyze Obsidian vault structure using pathlib and regex.

    Extracts wikilinks, computes stats, and finds orphaned notes
    without requiring obsidiantools.
    """

    def __init__(self, vault_path: Path) -> None:
        self.vault_path = vault_path.resolve()
        self._notes: list[Path] = []
        self._links: dict[str, list[str]] = {}
        self._scanned = False

    def _scan(self) -> None:
        """Scan vault and extract wikilinks from all notes."""
        if self._scanned:
            return

        self._notes = discover_notes(self.vault_path)
        for note_path in self._notes:
            try:
                content = note_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                logger.debug("skip_unreadable", path=str(note_path))
                continue

            stem = note_path.stem
            links = _WIKILINK_RE.findall(content)
            self._links[stem] = [link.strip() for link in links]

        self._scanned = True
        logger.info("vault_scanned", notes=len(self._notes), vault=str(self.vault_path))

    def get_wikilinks(self, path: Path) -> list[str]:
        """Get wikilinks from a specific note.

        Args:
            path: Absolute path to the note.

        Returns:
            List of wikilink targets found in the note.
        """
        self._scan()
        return list(self._links.get(path.stem, []))

    def find_orphaned(self) -> list[Path]:
        """Find notes with no incoming or outgoing links.

        Returns:
            List of paths to orphaned notes.
        """
        self._scan()

        # Notes that are linked to by at least one other note
        linked_targets: set[str] = set()
        for targets in self._links.values():
            linked_targets.update(targets)

        orphaned: list[Path] = []
        for note_path in self._notes:
            stem = note_path.stem
            has_outgoing = bool(self._links.get(stem))
            has_incoming = stem in linked_targets
            if not has_outgoing and not has_incoming:
                orphaned.append(note_path)

        return orphaned

    def analyze(self) -> VaultStats:
        """Compute comprehensive vault statistics.

        Returns:
            VaultStats with note counts, link counts, orphans, and broken links.
        """
        self._scan()

        all_stems = {p.stem for p in self._notes}

        # Count notes with frontmatter
        notes_with_fm = 0
        for note_path in self._notes:
            try:
                content = note_path.read_text(encoding="utf-8")
                if content.startswith("---"):
                    notes_with_fm += 1
            except (OSError, UnicodeDecodeError):
                continue

        # Count unique directories
        dirs = {p.parent for p in self._notes}

        # Total wikilinks
        total_links = sum(len(links) for links in self._links.values())

        # Broken links
        broken: list[tuple[str, str]] = []
        for stem, targets in self._links.items():
            for target in targets:
                if target not in all_stems:
                    broken.append((stem, target))

        # Orphaned notes
        linked_targets: set[str] = set()
        for targets in self._links.values():
            linked_targets.update(targets)

        orphaned: list[str] = []
        for note_path in self._notes:
            stem = note_path.stem
            if not self._links.get(stem) and stem not in linked_targets:
                orphaned.append(stem)

        return VaultStats(
            total_notes=len(self._notes),
            total_dirs=len(dirs),
            notes_with_frontmatter=notes_with_fm,
            wikilinks_count=total_links,
            orphaned_notes=tuple(orphaned),
            broken_links=tuple(broken),
        )

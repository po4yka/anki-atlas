"""Obsidian note parser: discovery and structured parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from pathlib import Path

import structlog

from packages.common.exceptions import ObsidianParseError
from packages.obsidian.frontmatter import parse_frontmatter

log = structlog.get_logger(__name__)

MAX_FILE_SIZE: Final[int] = 10 * 1024 * 1024  # 10 MB

_DEFAULT_IGNORE_DIRS: Final[tuple[str, ...]] = (".obsidian", ".trash", ".git")


@dataclass(frozen=True, slots=True)
class ParsedNote:
    """A parsed Obsidian markdown note."""

    path: Path
    frontmatter: dict[str, Any]
    content: str
    body: str
    sections: tuple[tuple[str, str], ...]
    title: str | None


def _extract_title(frontmatter: dict[str, Any], body: str) -> str | None:
    """Extract title from frontmatter or first H1 heading."""
    if frontmatter.get("title"):
        return str(frontmatter["title"])
    match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    return match.group(1).strip() if match else None


def _split_sections(body: str) -> tuple[tuple[str, str], ...]:
    """Split body into (heading, content) pairs by markdown headings.

    Content before the first heading gets an empty-string heading.
    """
    parts: list[tuple[str, str]] = []
    pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
    headings = list(pattern.finditer(body))

    if not headings:
        stripped = body.strip()
        if stripped:
            return (("", stripped),)
        return ()

    # Content before first heading
    pre = body[: headings[0].start()].strip()
    if pre:
        parts.append(("", pre))

    for i, heading_match in enumerate(headings):
        heading_text = heading_match.group(1).strip()
        start = heading_match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        section_content = body[start:end].strip()
        parts.append((heading_text, section_content))

    return tuple(parts)


def _validate_path(path: Path, vault_root: Path | None = None) -> Path:
    """Resolve and validate a file path.

    Checks existence, file type, size limit, and optional symlink traversal.
    """
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as e:
        msg = f"Cannot resolve path {path}: {e}"
        raise ObsidianParseError(msg) from e

    if not resolved.exists():
        msg = f"File does not exist: {resolved}"
        raise ObsidianParseError(msg)

    if not resolved.is_file():
        msg = f"Path is not a file: {resolved}"
        raise ObsidianParseError(msg)

    if vault_root is not None:
        resolved_root = vault_root.resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as e:
            msg = f"Path {resolved} is outside vault root {resolved_root}"
            raise ObsidianParseError(msg) from e

    file_size = resolved.stat().st_size
    if file_size > MAX_FILE_SIZE:
        msg = (
            f"File too large: {resolved} ({file_size} bytes). "
            f"Maximum allowed: {MAX_FILE_SIZE} bytes."
        )
        raise ObsidianParseError(msg)

    return resolved


def parse_note(path: Path, *, vault_root: Path | None = None) -> ParsedNote:
    """Parse a single Obsidian markdown note.

    Args:
        path: Path to the markdown file.
        vault_root: Optional vault root for symlink traversal protection.

    Returns:
        ParsedNote with extracted frontmatter, body, sections, and title.

    Raises:
        ObsidianParseError: If the file cannot be read or parsed.
    """
    resolved = _validate_path(path, vault_root)

    try:
        content = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        msg = f"Failed to read {resolved}: {e}"
        raise ObsidianParseError(msg) from e

    frontmatter = parse_frontmatter(content)

    # Strip frontmatter delimiters to get body
    fm_match = re.match(r"^---\s*\n.*?\n---\s*\n", content, re.DOTALL)
    body = content[fm_match.end() :] if fm_match else content

    title = _extract_title(frontmatter, body)
    sections = _split_sections(body)

    log.debug("parsed_note", path=str(resolved), title=title, sections=len(sections))

    return ParsedNote(
        path=resolved,
        frontmatter=frontmatter,
        content=content,
        body=body,
        sections=sections,
        title=title,
    )


def discover_notes(
    vault_root: Path,
    *,
    patterns: tuple[str, ...] = ("*.md",),
    ignore_dirs: tuple[str, ...] = _DEFAULT_IGNORE_DIRS,
) -> list[Path]:
    """Find all markdown notes in a vault.

    Args:
        vault_root: Root directory of the Obsidian vault.
        patterns: Glob patterns to match (default: all .md files).
        ignore_dirs: Directory names to skip.

    Returns:
        Sorted list of absolute paths to discovered notes.

    Raises:
        ObsidianParseError: If vault_root does not exist.
    """
    resolved_root = vault_root.resolve()
    if not resolved_root.is_dir():
        msg = f"Vault root is not a directory: {resolved_root}"
        raise ObsidianParseError(msg)

    notes: list[Path] = []
    for pattern in patterns:
        for md_path in resolved_root.rglob(pattern):
            # Skip ignored directories
            if any(part in ignore_dirs for part in md_path.parts):
                continue

            # Symlink traversal protection
            try:
                resolved = md_path.resolve()
                resolved.relative_to(resolved_root)
            except (ValueError, OSError):
                log.debug("skipping_outside_vault", path=str(md_path))
                continue

            if resolved.is_file():
                notes.append(resolved)

    notes.sort()
    log.info("discovered_notes", count=len(notes), vault=str(resolved_root))
    return notes

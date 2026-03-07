"""CLI command: generate flashcards from an Obsidian note."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console

from packages.common.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(module=__name__)
console = Console()


def generate(
    input_file: Path = typer.Argument(..., help="Path to an Obsidian markdown note"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without generating"),
) -> None:
    """Parse an Obsidian note and preview card generation."""
    from packages.obsidian.parser import parse_note

    resolved = input_file.expanduser().resolve()
    if not resolved.exists():
        console.print(f"[red]Error:[/red] File not found: {resolved}")
        raise typer.Exit(1)

    try:
        note = parse_note(resolved)
    except Exception as e:
        logger.error("generate.parse_failed", path=str(resolved), error=str(e))
        console.print(f"[red]Parse error:[/red] {e}")
        raise typer.Exit(1) from None

    console.print(f"[cyan]Note:[/cyan] {note.title or resolved.name}")
    console.print(f"  Sections: {len(note.sections)}")
    console.print(f"  Frontmatter keys: {', '.join(note.frontmatter.keys()) or 'none'}")

    if note.sections:
        console.print("\n[bold]Sections:[/bold]")
        for heading, content in note.sections:
            label = heading or "(preamble)"
            preview = content[:80] + "..." if len(content) > 80 else content
            console.print(f"  {label}: {preview}")

    if dry_run:
        console.print("\n[yellow]Dry run -- no cards generated[/yellow]")
    else:
        console.print(
            "\n[yellow]Card generation requires an LLM provider. "
            "Use the MCP tools or API for full generation.[/yellow]"
        )

"""CLI command: obsidian vault sync workflow."""

from pathlib import Path

import typer
from rich.console import Console

from packages.common.logging import get_logger

logger = get_logger(module=__name__)
console = Console()


def obsidian_sync(
    vault: Path = typer.Argument(..., help="Path to Obsidian vault"),
    source_dirs: str | None = typer.Option(
        None, "--source-dirs", "-s", help="Comma-separated subdirectories to scan"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Scan only, do not generate/sync"),
) -> None:
    """Scan an Obsidian vault and preview or sync cards."""
    from packages.obsidian import discover_notes

    vault_path = vault.expanduser().resolve()
    if not vault_path.is_dir():
        console.print(f"[red]Error:[/red] Vault not found: {vault_path}")
        raise typer.Exit(1)

    dirs = [d.strip() for d in source_dirs.split(",")] if source_dirs else None

    if dirs:
        all_notes: list[Path] = []
        for d in dirs:
            dir_path = vault_path / d
            if dir_path.is_dir():
                all_notes.extend(discover_notes(dir_path))
            else:
                console.print(f"[yellow]Warning:[/yellow] Directory not found: {dir_path}")
    else:
        all_notes = discover_notes(vault_path)

    console.print(f"[cyan]Vault:[/cyan] {vault_path}")
    console.print(f"  Notes discovered: {len(all_notes)}")

    if dry_run or not all_notes:
        if not all_notes:
            console.print("[yellow]No notes found[/yellow]")
        else:
            console.print("\n[bold]Discovered notes:[/bold]")
            for note_path in all_notes[:20]:
                rel = note_path.relative_to(vault_path) if vault_path in note_path.parents else note_path.name
                console.print(f"  {rel}")
            if len(all_notes) > 20:
                console.print(f"  ... and {len(all_notes) - 20} more")
            console.print("\n[yellow]Dry run -- no cards generated[/yellow]")
        return

    console.print(
        "\n[yellow]Full sync requires an LLM provider for card generation. "
        "Use --dry-run to preview, or use the MCP tools for full generation.[/yellow]"
    )

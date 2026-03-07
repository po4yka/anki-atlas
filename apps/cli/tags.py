"""CLI command: tag audit and normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from packages.common.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(module=__name__)
console = Console()


def tag_audit(
    input_file: Path = typer.Argument(..., help="File with tags, one per line"),
    fix: bool = typer.Option(False, "--fix", "-f", help="Show normalized tags"),
) -> None:
    """Audit tags for convention violations and suggest fixes."""
    from packages.taxonomy import normalize_tag, suggest_tag, validate_tag

    resolved = input_file.expanduser().resolve()
    if not resolved.exists():
        console.print(f"[red]Error:[/red] File not found: {resolved}")
        raise typer.Exit(1)

    tags = [
        line.strip() for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    if not tags:
        console.print("[yellow]No tags found in file[/yellow]")
        return

    valid_count = 0
    violation_count = 0

    table = Table(title=f"Tag Audit ({len(tags)} tags)")
    table.add_column("Tag", style="cyan")
    table.add_column("Status", width=6)
    if fix:
        table.add_column("Normalized")
    table.add_column("Issues")

    for tag in tags:
        issues = validate_tag(tag)
        if issues:
            violation_count += 1
            status = "[red]FAIL[/red]"
            issues_str = "; ".join(issues)
        else:
            valid_count += 1
            status = "[green]OK[/green]"
            issues_str = "-"

        if fix:
            normalized = normalize_tag(tag)
            norm_display = normalized if normalized != tag else "-"
            table.add_row(tag, status, norm_display, issues_str)
        else:
            table.add_row(tag, status, issues_str)

    console.print(table)

    # Summary
    console.print(f"\n[bold]Summary:[/bold] {valid_count} valid, {violation_count} violations")

    # Suggestions for invalid tags
    if violation_count > 0:
        invalid_tags = [t for t in tags if validate_tag(t)]
        suggestions_shown = 0
        for tag in invalid_tags:
            matches = suggest_tag(tag)
            if matches and suggestions_shown < 10:
                console.print(f"  [dim]{tag}[/dim] -> {', '.join(matches[:3])}")
                suggestions_shown += 1

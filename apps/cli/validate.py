"""CLI command: validate flashcard content."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from packages.common.logging import get_logger

logger = get_logger(module=__name__)
console = Console()


def validate(
    input_file: Path = typer.Argument(..., help="File with card front/back (--- separated)"),
    quality: bool = typer.Option(False, "--quality", "-q", help="Run quality assessment"),
) -> None:
    """Validate flashcard content from a file.

    The file should contain front and back separated by a line with just '---'.
    Multiple cards can be separated by blank lines followed by another front/back pair.
    """
    from packages.validation import (
        ContentValidator,
        FormatValidator,
        TagValidator,
        ValidationPipeline,
        assess_quality,
    )

    resolved = input_file.expanduser().resolve()
    if not resolved.exists():
        console.print(f"[red]Error:[/red] File not found: {resolved}")
        raise typer.Exit(1)

    text = resolved.read_text(encoding="utf-8")
    cards = _parse_card_file(text)

    if not cards:
        console.print("[red]Error:[/red] No cards found. Use '---' to separate front/back.")
        raise typer.Exit(1)

    pipeline = ValidationPipeline([ContentValidator(), FormatValidator(), TagValidator()])

    table = Table(title=f"Validation Results ({len(cards)} cards)")
    table.add_column("#", style="dim", width=3)
    table.add_column("Status", width=6)
    table.add_column("Issues")

    has_errors = False
    for i, (front, back) in enumerate(cards, 1):
        result = pipeline.run(front=front, back=back)
        status = "[green]PASS[/green]" if result.is_valid else "[red]FAIL[/red]"
        if not result.is_valid:
            has_errors = True

        issues_str = "; ".join(f"{iss.severity}: {iss.message}" for iss in result.issues)
        table.add_row(str(i), status, issues_str or "-")

    console.print(table)

    if quality:
        console.print("\n[bold]Quality Scores:[/bold]")
        for i, (front, back) in enumerate(cards, 1):
            score = assess_quality(front=front, back=back)
            console.print(
                f"  Card {i}: overall={score.overall:.2f} "
                f"(clarity={score.clarity:.1f} atom={score.atomicity:.1f} "
                f"test={score.testability:.1f} memo={score.memorability:.1f} "
                f"acc={score.accuracy:.1f})"
            )

    if has_errors:
        raise typer.Exit(1)


def _parse_card_file(text: str) -> list[tuple[str, str]]:
    """Parse a file into (front, back) pairs split by '---'."""
    cards: list[tuple[str, str]] = []
    # Split on double newlines for multiple cards
    blocks = text.strip().split("\n\n\n")
    for block in blocks:
        block = block.strip()
        if "---" in block:
            parts = block.split("---", maxsplit=1)
            front = parts[0].strip()
            back = parts[1].strip() if len(parts) > 1 else ""
            if front or back:
                cards.append((front, back))
    return cards

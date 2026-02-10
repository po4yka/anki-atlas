"""Anki Atlas CLI application."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="anki-atlas",
    help="Searchable hybrid index for Anki collections",
    no_args_is_help=True,
)

console = Console()


@app.command()
def version() -> None:
    """Show version information."""
    console.print("anki-atlas 0.1.0")


@app.command()
def sync(
    source: str = typer.Option(
        ...,
        "--source",
        "-s",
        help="Path to collection.anki2 file",
    ),
    run_migrations: bool = typer.Option(
        True,
        "--migrate/--no-migrate",
        help="Run database migrations before sync",
    ),
) -> None:
    """Sync Anki collection to the index."""
    asyncio.run(_sync_async(source, run_migrations))


async def _sync_async(source: str, run_migrations: bool) -> None:
    """Async sync implementation."""
    from packages.anki.sync import sync_anki_collection
    from packages.common.database import run_migrations as db_migrate

    source_path = Path(source).expanduser().resolve()

    if not source_path.exists():
        console.print(f"[red]Error:[/red] Collection not found: {source_path}")
        raise typer.Exit(1)

    console.print(f"Syncing from: {source_path}")

    # Run migrations if requested
    if run_migrations:
        console.print("Running database migrations...")
        try:
            applied = await db_migrate()
            if applied:
                console.print(f"[green]Applied migrations:[/green] {', '.join(applied)}")
        except Exception as e:
            console.print(f"[red]Migration error:[/red] {e}")
            raise typer.Exit(1) from None

    # Run sync
    console.print("Syncing collection...")
    try:
        stats = await sync_anki_collection(source_path)

        # Display results
        table = Table(title="Sync Complete")
        table.add_column("Entity", style="cyan")
        table.add_column("Count", justify="right", style="green")

        table.add_row("Decks", str(stats.decks_upserted))
        table.add_row("Models", str(stats.models_upserted))
        table.add_row("Notes", str(stats.notes_upserted))
        table.add_row("Notes deleted", str(stats.notes_deleted))
        table.add_row("Cards", str(stats.cards_upserted))
        table.add_row("Card stats", str(stats.card_stats_upserted))
        table.add_row("Duration (ms)", str(stats.duration_ms))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Sync error:[/red] {e}")
        raise typer.Exit(1) from None


@app.command()
def migrate() -> None:
    """Run database migrations."""
    asyncio.run(_migrate_async())


async def _migrate_async() -> None:
    """Async migrate implementation."""
    from packages.common.database import run_migrations

    console.print("Running database migrations...")
    try:
        applied = await run_migrations()
        if applied:
            console.print(f"[green]Applied migrations:[/green] {', '.join(applied)}")
        else:
            console.print("[yellow]No migrations to apply[/yellow]")
    except Exception as e:
        console.print(f"[red]Migration error:[/red] {e}")
        raise typer.Exit(1) from None


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    deck: str | None = typer.Option(None, "--deck", "-d", help="Filter by deck"),
    tag: str | None = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    top: int = typer.Option(10, "--top", "-n", help="Number of results"),
) -> None:
    """Search the Anki index."""
    console.print(f"Searching: {query} (deck={deck}, tag={tag}, top={top})")
    console.print("[yellow]Not implemented yet[/yellow]")
    raise typer.Exit(1)


@app.command()
def coverage(
    topic: str = typer.Argument(..., help="Topic path (e.g., android/compose/state)"),
) -> None:
    """Show topic coverage metrics."""
    console.print(f"Coverage for: {topic}")
    console.print("[yellow]Not implemented yet[/yellow]")
    raise typer.Exit(1)


@app.command()
def gaps(
    topic: str = typer.Argument(..., help="Topic path"),
    min_coverage: int = typer.Option(5, "--min-coverage", help="Minimum coverage threshold"),
) -> None:
    """Detect gaps in topic coverage."""
    console.print(f"Gaps for: {topic} (min_coverage={min_coverage})")
    console.print("[yellow]Not implemented yet[/yellow]")
    raise typer.Exit(1)


@app.command()
def duplicates(
    threshold: float = typer.Option(0.92, "--threshold", help="Similarity threshold"),
) -> None:
    """Find duplicate cards."""
    console.print(f"Finding duplicates (threshold={threshold})")
    console.print("[yellow]Not implemented yet[/yellow]")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()

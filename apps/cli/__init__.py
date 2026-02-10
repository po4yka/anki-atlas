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
    index: bool = typer.Option(
        True,
        "--index/--no-index",
        help="Index notes to vector database after sync",
    ),
    force_reindex: bool = typer.Option(
        False,
        "--force-reindex",
        help="Force re-embedding all notes",
    ),
) -> None:
    """Sync Anki collection to the index."""
    asyncio.run(_sync_async(source, run_migrations, index, force_reindex))


async def _sync_async(
    source: str,
    run_migrations: bool,
    index: bool,
    force_reindex: bool,
) -> None:
    """Async sync implementation."""
    from packages.anki.sync import sync_anki_collection
    from packages.common.database import run_migrations as db_migrate
    from packages.indexer.service import index_all_notes

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

    # Run sync to PostgreSQL
    console.print("Syncing collection to PostgreSQL...")
    try:
        sync_stats = await sync_anki_collection(source_path)

        table = Table(title="PostgreSQL Sync")
        table.add_column("Entity", style="cyan")
        table.add_column("Count", justify="right", style="green")

        table.add_row("Decks", str(sync_stats.decks_upserted))
        table.add_row("Models", str(sync_stats.models_upserted))
        table.add_row("Notes", str(sync_stats.notes_upserted))
        table.add_row("Notes deleted", str(sync_stats.notes_deleted))
        table.add_row("Cards", str(sync_stats.cards_upserted))
        table.add_row("Card stats", str(sync_stats.card_stats_upserted))
        table.add_row("Duration (ms)", str(sync_stats.duration_ms))

        console.print(table)

    except Exception as e:
        console.print(f"[red]Sync error:[/red] {e}")
        raise typer.Exit(1) from None

    # Run indexing if requested
    if index:
        console.print("\nIndexing notes to Qdrant...")
        try:
            index_stats = await index_all_notes(force_reindex=force_reindex)

            table = Table(title="Vector Index")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", justify="right", style="green")

            table.add_row("Notes processed", str(index_stats.notes_processed))
            table.add_row("Notes embedded", str(index_stats.notes_embedded))
            table.add_row("Notes skipped", str(index_stats.notes_skipped))
            table.add_row("Notes deleted", str(index_stats.notes_deleted))

            console.print(table)

            if index_stats.errors:
                for error in index_stats.errors:
                    console.print(f"[yellow]Warning:[/yellow] {error}")

        except Exception as e:
            console.print(f"[red]Indexing error:[/red] {e}")
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
def index(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-embedding all notes",
    ),
) -> None:
    """Index notes from PostgreSQL to vector database."""
    asyncio.run(_index_async(force))


async def _index_async(force: bool) -> None:
    """Async index implementation."""
    from packages.indexer.service import index_all_notes

    console.print("Indexing notes to Qdrant...")
    try:
        stats = await index_all_notes(force_reindex=force)

        table = Table(title="Vector Index")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")

        table.add_row("Notes processed", str(stats.notes_processed))
        table.add_row("Notes embedded", str(stats.notes_embedded))
        table.add_row("Notes skipped", str(stats.notes_skipped))
        table.add_row("Notes deleted", str(stats.notes_deleted))

        console.print(table)

        if stats.errors:
            for error in stats.errors:
                console.print(f"[yellow]Warning:[/yellow] {error}")

    except Exception as e:
        console.print(f"[red]Indexing error:[/red] {e}")
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

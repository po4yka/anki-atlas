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
    deck: str | None = typer.Option(None, "--deck", "-d", help="Filter by deck name"),
    tag: str | None = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    top: int = typer.Option(10, "--top", "-n", help="Number of results"),
    semantic_only: bool = typer.Option(False, "--semantic", help="Use only semantic search"),
    fts_only: bool = typer.Option(False, "--fts", help="Use only full-text search"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed scores"),
) -> None:
    """Search the Anki index."""
    asyncio.run(_search_async(query, deck, tag, top, semantic_only, fts_only, verbose))


async def _search_async(
    query: str,
    deck: str | None,
    tag: str | None,
    top: int,
    semantic_only: bool,
    fts_only: bool,
    verbose: bool,
) -> None:
    """Async search implementation."""
    from packages.search import SearchFilters, SearchService

    filters = SearchFilters(
        deck_names=[deck] if deck else None,
        tags=[tag] if tag else None,
    )

    service = SearchService()

    console.print(f"Searching for: [cyan]{query}[/cyan]")
    if deck:
        console.print(f"  Deck filter: {deck}")
    if tag:
        console.print(f"  Tag filter: {tag}")

    try:
        result = await service.search(
            query=query,
            filters=filters,
            limit=top,
            semantic_only=semantic_only,
            fts_only=fts_only,
        )

        if not result.results:
            console.print("[yellow]No results found[/yellow]")
            return

        # Fetch note details
        note_ids = [r.note_id for r in result.results]
        notes_details = await service.get_notes_details(note_ids)

        # Display results
        table = Table(title=f"Search Results ({len(result.results)} found)")
        table.add_column("#", style="dim", width=3)
        table.add_column("Note ID", style="cyan")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Sources", style="magenta")
        table.add_column("Preview", style="white", no_wrap=False, max_width=60)

        for i, r in enumerate(result.results, 1):
            detail = notes_details.get(r.note_id)
            preview = ""
            if r.headline:
                # Use FTS headline if available
                preview = r.headline.replace("<<", "[bold]").replace(">>", "[/bold]")
            elif detail:
                # Fallback to truncated normalized text
                preview = detail.normalized_text[:100] + "..." if len(detail.normalized_text) > 100 else detail.normalized_text

            sources = ", ".join(r.sources)
            score = f"{r.rrf_score:.4f}"

            table.add_row(str(i), str(r.note_id), score, sources, preview)

        console.print(table)

        # Show stats
        console.print(f"\n[dim]Stats: {result.stats.both} in both, "
                      f"{result.stats.semantic_only} semantic only, "
                      f"{result.stats.fts_only} FTS only[/dim]")

        # Verbose mode: show detailed scores
        if verbose and result.results:
            console.print("\n[bold]Detailed Scores:[/bold]")
            for r in result.results[:5]:  # Top 5 only
                detail = notes_details.get(r.note_id)
                tags_str = ", ".join(detail.tags) if detail and detail.tags else "none"
                console.print(f"  Note {r.note_id}:")
                console.print(f"    RRF: {r.rrf_score:.4f}")
                if r.semantic_score is not None:
                    console.print(f"    Semantic: {r.semantic_score:.4f} (rank {r.semantic_rank})")
                if r.fts_score is not None:
                    console.print(f"    FTS: {r.fts_score:.4f} (rank {r.fts_rank})")
                console.print(f"    Tags: {tags_str}")

    except Exception as e:
        console.print(f"[red]Search error:[/red] {e}")
        raise typer.Exit(1) from None


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

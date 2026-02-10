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
def topics(
    taxonomy_file: str | None = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to topics.yml file to load",
    ),
    label: bool = typer.Option(
        False,
        "--label",
        "-l",
        help="Label notes with topics after loading",
    ),
    min_confidence: float = typer.Option(
        0.3,
        "--min-confidence",
        help="Minimum confidence for labeling",
    ),
) -> None:
    """Manage topic taxonomy."""
    asyncio.run(_topics_async(taxonomy_file, label, min_confidence))


async def _topics_async(
    taxonomy_file: str | None,
    label: bool,
    min_confidence: float,
) -> None:
    """Async topics implementation."""
    from pathlib import Path

    from packages.analytics import AnalyticsService

    service = AnalyticsService()

    try:
        # Load taxonomy
        if taxonomy_file:
            yaml_path = Path(taxonomy_file).expanduser().resolve()
            if not yaml_path.exists():
                console.print(f"[red]File not found:[/red] {yaml_path}")
                raise typer.Exit(1) from None

            console.print(f"Loading taxonomy from: {yaml_path}")
            taxonomy = await service.load_taxonomy(yaml_path)
            console.print(f"[green]Loaded {len(taxonomy.topics)} topics[/green]")
        else:
            taxonomy = await service.load_taxonomy()
            console.print(f"Loaded {len(taxonomy.topics)} topics from database")

        # Display taxonomy tree
        tree = await service.get_taxonomy_tree()

        if tree:
            table = Table(title="Topic Taxonomy")
            table.add_column("Path", style="cyan")
            table.add_column("Label")
            table.add_column("Notes", justify="right", style="green")
            table.add_column("Mature", justify="right")

            for t in tree:
                indent = "  " * t["depth"]
                table.add_row(
                    indent + t["path"].split("/")[-1],
                    t["label"],
                    str(t["note_count"]),
                    str(t["mature_count"]),
                )

            console.print(table)
        else:
            console.print("[yellow]No topics in database[/yellow]")

        # Label notes if requested
        if label and taxonomy.topics:
            console.print("\nLabeling notes with topics...")
            stats = await service.label_notes(taxonomy, min_confidence)

            console.print("[green]Labeling complete:[/green]")
            console.print(f"  Notes processed: {stats.notes_processed}")
            console.print(f"  Assignments created: {stats.assignments_created}")
            console.print(f"  Topics matched: {stats.topics_matched}")

    except Exception as e:
        console.print(f"[red]Topics error:[/red] {e}")
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
    topic: str = typer.Argument(..., help="Topic path (e.g., programming/python)"),
    include_subtree: bool = typer.Option(True, "--subtree/--no-subtree", help="Include child topics"),
) -> None:
    """Show topic coverage metrics."""
    asyncio.run(_coverage_async(topic, include_subtree))


async def _coverage_async(topic: str, include_subtree: bool) -> None:
    """Async coverage implementation."""
    from packages.analytics import AnalyticsService

    service = AnalyticsService()

    console.print(f"Coverage for: [cyan]{topic}[/cyan]")
    if include_subtree:
        console.print("  (including subtree)")

    try:
        cov = await service.get_coverage(topic, include_subtree)

        if not cov:
            console.print(f"[red]Topic not found:[/red] {topic}")
            raise typer.Exit(1) from None

        table = Table(title=f"Coverage: {cov.label}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Notes", str(cov.note_count))
        table.add_row("Subtree notes", str(cov.subtree_count))
        table.add_row("Mature notes", str(cov.mature_count))
        table.add_row("Child topics", str(cov.child_count))
        table.add_row("Covered children", str(cov.covered_children))
        table.add_row("Avg confidence", f"{cov.avg_confidence:.2%}")
        table.add_row("Weak notes", str(cov.weak_notes))
        table.add_row("Avg lapses", f"{cov.avg_lapses:.1f}")

        console.print(table)

        # Show breadth coverage
        if cov.child_count > 0:
            breadth = cov.covered_children / cov.child_count
            console.print(f"\n[dim]Breadth coverage: {breadth:.0%} ({cov.covered_children}/{cov.child_count} children covered)[/dim]")

    except Exception as e:
        console.print(f"[red]Coverage error:[/red] {e}")
        raise typer.Exit(1) from None


@app.command()
def gaps(
    topic: str = typer.Argument(..., help="Topic path"),
    min_coverage: int = typer.Option(1, "--min-coverage", "-m", help="Minimum notes for coverage"),
) -> None:
    """Detect gaps in topic coverage."""
    asyncio.run(_gaps_async(topic, min_coverage))


async def _gaps_async(topic: str, min_coverage: int) -> None:
    """Async gaps implementation."""
    from packages.analytics import AnalyticsService

    service = AnalyticsService()

    console.print(f"Gaps for: [cyan]{topic}[/cyan] (min_coverage={min_coverage})")

    try:
        gaps_list = await service.get_gaps(topic, min_coverage)

        if not gaps_list:
            console.print("[green]No gaps found - all topics have sufficient coverage[/green]")
            return

        # Separate missing and undercovered
        missing = [g for g in gaps_list if g.gap_type == "missing"]
        undercovered = [g for g in gaps_list if g.gap_type == "undercovered"]

        if missing:
            table = Table(title=f"Missing Topics ({len(missing)})")
            table.add_column("Path", style="red")
            table.add_column("Label")
            table.add_column("Description", max_width=40)

            for g in missing:
                table.add_row(g.path, g.label, g.description or "")

            console.print(table)

        if undercovered:
            table = Table(title=f"Undercovered Topics ({len(undercovered)})")
            table.add_column("Path", style="yellow")
            table.add_column("Label")
            table.add_column("Notes", justify="right")
            table.add_column("Threshold", justify="right")

            for g in undercovered:
                table.add_row(g.path, g.label, str(g.note_count), str(g.threshold))

            console.print(table)

        # Summary
        console.print(f"\n[dim]Total gaps: {len(gaps_list)} ({len(missing)} missing, {len(undercovered)} undercovered)[/dim]")

    except Exception as e:
        console.print(f"[red]Gaps error:[/red] {e}")
        raise typer.Exit(1) from None


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

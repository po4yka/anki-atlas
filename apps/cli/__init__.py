"""Anki Atlas CLI application."""

import typer

app = typer.Typer(
    name="anki-atlas",
    help="Searchable hybrid index for Anki collections",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Show version information."""
    typer.echo("anki-atlas 0.1.0")


@app.command()
def sync(
    source: str = typer.Option(
        None,
        "--source",
        "-s",
        help="Path to collection.anki2 file",
    ),
) -> None:
    """Sync Anki collection to the index."""
    typer.echo(f"Syncing from: {source}")
    typer.echo("Not implemented yet")
    raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    deck: str | None = typer.Option(None, "--deck", "-d", help="Filter by deck"),
    tag: str | None = typer.Option(None, "--tag", "-t", help="Filter by tag"),
    top: int = typer.Option(10, "--top", "-n", help="Number of results"),
) -> None:
    """Search the Anki index."""
    typer.echo(f"Searching: {query} (deck={deck}, tag={tag}, top={top})")
    typer.echo("Not implemented yet")
    raise typer.Exit(1)


@app.command()
def coverage(
    topic: str = typer.Argument(..., help="Topic path (e.g., android/compose/state)"),
) -> None:
    """Show topic coverage metrics."""
    typer.echo(f"Coverage for: {topic}")
    typer.echo("Not implemented yet")
    raise typer.Exit(1)


@app.command()
def gaps(
    topic: str = typer.Argument(..., help="Topic path"),
    min_coverage: int = typer.Option(5, "--min-coverage", help="Minimum coverage threshold"),
) -> None:
    """Detect gaps in topic coverage."""
    typer.echo(f"Gaps for: {topic} (min_coverage={min_coverage})")
    typer.echo("Not implemented yet")
    raise typer.Exit(1)


@app.command()
def duplicates(
    threshold: float = typer.Option(0.92, "--threshold", help="Similarity threshold"),
) -> None:
    """Find duplicate cards."""
    typer.echo(f"Finding duplicates (threshold={threshold})")
    typer.echo("Not implemented yet")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()

"""Markdown formatters for MCP tool responses."""

from typing import Any


def format_search_result(
    result: Any,
    note_details: dict[int, Any],
) -> str:
    """Format search results as markdown.

    Args:
        result: Hybrid search result.
        note_details: Mapping of note_id to note details.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []

    lines.append(f"## Search Results for: {result.query}")
    lines.append("")

    if not result.results:
        lines.append("No results found.")
        return "\n".join(lines)

    # Stats summary
    stats = result.stats
    lines.append(
        f"Found **{stats.total}** results "
        f"(semantic only: {stats.semantic_only}, FTS only: {stats.fts_only}, both: {stats.both})"
    )
    lines.append("")

    # Filters applied
    if result.filters_applied:
        filters_str = ", ".join(f"{k}={v}" for k, v in result.filters_applied.items())
        lines.append(f"*Filters: {filters_str}*")
        lines.append("")

    # Results table
    lines.append("| # | Score | Note ID | Preview | Tags | Deck |")
    lines.append("|---|-------|---------|---------|------|------|")

    for i, sr in enumerate(result.results[:20], 1):  # Limit to 20 for readability
        detail = note_details.get(sr.note_id)
        if detail:
            preview = _truncate(detail.normalized_text, 60)
            tags = ", ".join(detail.tags[:3]) if detail.tags else "-"
            deck = detail.deck_names[0] if detail.deck_names else "-"
        else:
            preview = sr.headline or "-"
            tags = "-"
            deck = "-"

        sources = "+".join(sr.sources) if sr.sources else "-"
        score_label = f"{sr.rrf_score:.3f}"
        if getattr(sr, "rerank_score", None) is not None:
            score_label = f"{score_label} / CE {sr.rerank_score:.3f}"
        lines.append(
            f"| {i} | {score_label} ({sources}) | {sr.note_id} | {preview} | {tags} | {deck} |"
        )

    if len(result.results) > 20:
        lines.append(f"\n*...and {len(result.results) - 20} more results*")

    return "\n".join(lines)


def format_coverage_result(coverage: Any) -> str:
    """Format topic coverage as markdown.

    Args:
        coverage: Topic coverage metrics.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []

    lines.append(f"## Coverage: {coverage.label}")
    lines.append(f"*Path: {coverage.path}*")
    lines.append("")

    # Core metrics
    lines.append("### Note Counts")
    lines.append(f"- **Direct notes**: {coverage.note_count}")
    lines.append(f"- **Subtree total**: {coverage.subtree_count}")
    lines.append(f"- **Mature notes**: {coverage.mature_count}")
    lines.append("")

    # Child coverage
    if coverage.child_count > 0:
        coverage_pct = (coverage.covered_children / coverage.child_count) * 100
        lines.append("### Child Topics")
        lines.append(
            f"- **Covered**: {coverage.covered_children}/{coverage.child_count} "
            f"({coverage_pct:.0f}%)"
        )
        lines.append("")

    # Quality metrics
    lines.append("### Quality Metrics")
    lines.append(f"- **Avg confidence**: {coverage.avg_confidence:.2f}")
    lines.append(f"- **Avg lapses**: {coverage.avg_lapses:.1f}")
    lines.append(f"- **Weak notes**: {coverage.weak_notes}")

    return "\n".join(lines)


def format_gaps_result(gaps: list[Any], topic_path: str) -> str:
    """Format topic gaps as markdown.

    Args:
        gaps: List of topic gaps.
        topic_path: Root topic path analyzed.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []

    lines.append(f"## Knowledge Gaps: {topic_path}")
    lines.append("")

    if not gaps:
        lines.append("No gaps found. All topics have sufficient coverage.")
        return "\n".join(lines)

    # Separate missing vs undercovered
    missing = [g for g in gaps if g.gap_type == "missing"]
    undercovered = [g for g in gaps if g.gap_type == "undercovered"]

    if missing:
        lines.append(f"### Missing Topics ({len(missing)})")
        lines.append("*Topics with zero notes*")
        lines.append("")
        lines.append("| Topic | Path | Description |")
        lines.append("|-------|------|-------------|")
        for gap in missing[:15]:
            desc = _truncate(gap.description or "-", 40)
            lines.append(f"| {gap.label} | `{gap.path}` | {desc} |")
        if len(missing) > 15:
            lines.append(f"\n*...and {len(missing) - 15} more missing topics*")
        lines.append("")

    if undercovered:
        lines.append(f"### Undercovered Topics ({len(undercovered)})")
        lines.append(f"*Topics with fewer than {undercovered[0].threshold} notes*")
        lines.append("")
        lines.append("| Topic | Path | Notes | Threshold |")
        lines.append("|-------|------|-------|-----------|")
        for gap in undercovered[:15]:
            lines.append(f"| {gap.label} | `{gap.path}` | {gap.note_count} | {gap.threshold} |")
        if len(undercovered) > 15:
            lines.append(f"\n*...and {len(undercovered) - 15} more undercovered topics*")

    return "\n".join(lines)


def format_duplicates_result(
    clusters: list[Any],
    stats: Any,
) -> str:
    """Format duplicate detection results as markdown.

    Args:
        clusters: List of duplicate clusters.
        stats: Duplicate detection statistics.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []

    lines.append("## Duplicate Detection Results")
    lines.append("")

    # Stats summary
    lines.append("### Summary")
    lines.append(f"- **Notes scanned**: {stats.notes_scanned}")
    lines.append(f"- **Clusters found**: {stats.clusters_found}")
    lines.append(f"- **Total duplicates**: {stats.total_duplicates}")
    lines.append(f"- **Avg cluster size**: {stats.avg_cluster_size:.1f}")
    lines.append("")

    if not clusters:
        lines.append("No duplicate clusters found.")
        return "\n".join(lines)

    # Cluster details
    lines.append("### Duplicate Clusters")
    lines.append("")

    for i, cluster in enumerate(clusters[:10], 1):  # Limit to 10
        lines.append(f"#### Cluster {i} ({cluster.size} notes)")
        lines.append(f"**Representative** (note {cluster.representative_id}):")
        lines.append(f"> {_truncate(cluster.representative_text, 150)}")
        lines.append("")

        if cluster.duplicates:
            lines.append("**Duplicates:**")
            for dup in cluster.duplicates[:5]:
                sim_pct = dup["similarity"] * 100
                lines.append(
                    f"- Note {dup['note_id']} ({sim_pct:.0f}% similar): "
                    f"{_truncate(dup['text'], 80)}"
                )
            if len(cluster.duplicates) > 5:
                lines.append(f"- *...and {len(cluster.duplicates) - 5} more*")
        lines.append("")

    if len(clusters) > 10:
        lines.append(f"*...and {len(clusters) - 10} more clusters*")

    return "\n".join(lines)


def format_sync_result(stats: Any) -> str:
    """Format sync results as markdown.

    Args:
        stats: Sync operation statistics.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []

    lines.append("## Sync Complete")
    lines.append("")

    duration_sec = stats.duration_ms / 1000
    lines.append(f"*Duration: {duration_sec:.1f}s*")
    lines.append("")

    lines.append("### Records Synced")
    lines.append(f"- **Decks**: {stats.decks_upserted}")
    lines.append(f"- **Models**: {stats.models_upserted}")
    lines.append(f"- **Notes**: {stats.notes_upserted}")
    lines.append(f"- **Cards**: {stats.cards_upserted}")
    lines.append(f"- **Card stats**: {stats.card_stats_upserted}")

    if stats.notes_deleted > 0:
        lines.append("")
        lines.append(f"*Notes removed: {stats.notes_deleted}*")

    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

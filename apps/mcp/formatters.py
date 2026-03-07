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


def format_generate_result(
    title: str | None,
    sections: tuple[tuple[str, str], ...],
    body_length: int,
) -> str:
    """Format note parse result as markdown for generation preview.

    Args:
        title: Extracted note title.
        sections: Parsed (heading, content) pairs.
        body_length: Length of the note body in characters.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []

    lines.append("## Generation Preview")
    lines.append("")

    if title:
        lines.append(f"**Title**: {title}")
    else:
        lines.append("**Title**: *(not detected)*")
    lines.append(f"**Body length**: {body_length} chars")
    lines.append(f"**Sections**: {len(sections)}")
    lines.append("")

    if sections:
        lines.append("### Sections Found")
        lines.append("")
        lines.append("| # | Heading | Content Length |")
        lines.append("|---|---------|---------------|")
        for i, (heading, content) in enumerate(sections, 1):
            heading_label = heading if heading else "*(preamble)*"
            lines.append(f"| {i} | {_truncate(heading_label, 50)} | {len(content)} chars |")
        lines.append("")

    estimated_cards = max(1, len(sections))
    lines.append(f"**Estimated cards**: ~{estimated_cards}")
    lines.append("")
    lines.append("*Use the generate command to create cards from this note.*")

    return "\n".join(lines)


def format_validate_result(
    result: Any,
    quality: Any | None = None,
) -> str:
    """Format validation result as markdown.

    Args:
        result: ValidationResult from the pipeline.
        quality: Optional QualityScore.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []

    status = "PASS" if result.is_valid else "FAIL"
    lines.append(f"## Validation: {status}")
    lines.append("")

    errors = result.errors()
    warnings = result.warnings()

    lines.append(f"- **Errors**: {len(errors)}")
    lines.append(f"- **Warnings**: {len(warnings)}")
    lines.append("")

    if errors:
        lines.append("### Errors")
        for issue in errors:
            loc = f" ({issue.location})" if issue.location else ""
            lines.append(f"- {issue.message}{loc}")
        lines.append("")

    if warnings:
        lines.append("### Warnings")
        for issue in warnings:
            loc = f" ({issue.location})" if issue.location else ""
            lines.append(f"- {issue.message}{loc}")
        lines.append("")

    if quality is not None:
        lines.append("### Quality Score")
        lines.append(f"- **Overall**: {quality.overall:.2f}")
        lines.append(f"- Clarity: {quality.clarity:.2f}")
        lines.append(f"- Atomicity: {quality.atomicity:.2f}")
        lines.append(f"- Testability: {quality.testability:.2f}")
        lines.append(f"- Memorability: {quality.memorability:.2f}")
        lines.append(f"- Accuracy: {quality.accuracy:.2f}")

    return "\n".join(lines)


def format_obsidian_sync_result(
    notes_found: int,
    parsed_notes: list[tuple[str, str | None, int]],
    vault_path: str,
) -> str:
    """Format obsidian sync discovery result as markdown.

    Args:
        notes_found: Total number of markdown files found.
        parsed_notes: List of (filename, title, section_count) tuples.
        vault_path: Path to the vault.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []

    lines.append("## Obsidian Vault Scan")
    lines.append(f"*Vault: {vault_path}*")
    lines.append("")
    lines.append(f"**Notes found**: {notes_found}")
    lines.append("")

    if parsed_notes:
        lines.append("### Parsed Notes")
        lines.append("")
        lines.append("| # | File | Title | Sections |")
        lines.append("|---|------|-------|----------|")
        for i, (filename, title, section_count) in enumerate(parsed_notes[:20], 1):
            title_label = title or "*(untitled)*"
            lines.append(
                f"| {i} | {_truncate(filename, 40)} | "
                f"{_truncate(title_label, 40)} | {section_count} |"
            )
        if len(parsed_notes) > 20:
            lines.append(f"\n*...and {len(parsed_notes) - 20} more notes*")

    return "\n".join(lines)


def format_tag_audit_result(
    results: list[tuple[str, list[str], str | None, list[str]]],
) -> str:
    """Format tag audit result as markdown.

    Args:
        results: List of (tag, issues, normalized, suggestions) tuples.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []

    lines.append("## Tag Audit Results")
    lines.append("")

    total = len(results)
    valid = sum(1 for _, issues, _, _ in results if not issues)
    invalid = total - valid

    lines.append(f"- **Total tags**: {total}")
    lines.append(f"- **Valid**: {valid}")
    lines.append(f"- **With issues**: {invalid}")
    lines.append("")

    if invalid > 0:
        lines.append("### Issues Found")
        lines.append("")
        lines.append("| Tag | Issues | Normalized | Suggestions |")
        lines.append("|-----|--------|------------|-------------|")
        for tag, issues, normalized, suggestions in results:
            if not issues:
                continue
            issues_str = "; ".join(issues)
            norm_str = normalized or "-"
            sugg_str = ", ".join(suggestions[:3]) if suggestions else "-"
            lines.append(
                f"| `{_truncate(tag, 30)}` | {_truncate(issues_str, 50)} | "
                f"`{norm_str}` | {sugg_str} |"
            )
        lines.append("")

    if valid == total:
        lines.append("All tags are valid.")

    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

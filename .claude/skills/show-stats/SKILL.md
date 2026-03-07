---
name: show-stats
description: Generates reports on sync status, card coverage, deck health, and vector database statistics. Use for system overview or progress tracking.
allowed-tools: Read, Bash
---

# Show Statistics and Reports

Generate user-friendly reports on your flashcard system status.

## When to Use

- User asks for "stats", "statistics", or "report"
- User wants "sync status" or "deck health"
- User asks "how many cards do I have?"
- User wants system "overview"

## Workflow

### Step 1: Determine Report Type

| Type | Command | Shows |
|------|---------|-------|
| Coverage | `coverage` | Topic coverage metrics |
| Gaps | `gaps` | Missing/undercovered topics |
| Duplicates | `duplicates` | Near-duplicate notes |
| Search | `search` | Search index health |

### Step 2: Gather Statistics

```bash
# Topic coverage
uv run anki-atlas coverage "programming"

# Coverage gaps
uv run anki-atlas gaps "programming" --min-coverage 1

# Find duplicates
uv run anki-atlas duplicates --threshold 0.92

# Test search health
uv run anki-atlas search "test query" --verbose
```

### Step 3: Present Report

```
Flashcard System Overview
=========================

Index Status:
- Total notes indexed
- Search health: OK
- Last sync: timestamp

Coverage:
- Full coverage: 25 topics
- Partial: 5 topics
- Gaps: 10 topics
- Orphan cards: 2

Use '/find-gaps' for detailed coverage.

By Deck:
- Kotlin: 80 cards
- Android: 45 cards
- Python: 25 cards

Duplicates:
- Clusters found: 3
- Total duplicate notes: 7
```

### Step 4: Actionable Insights

Based on stats, suggest:
- **Gaps found**: "10 topics uncovered. Run `/find-gaps`."
- **Orphans**: "2 orphans. Run `/cleanup-cards`."
- **Duplicates**: "3 duplicate clusters. Run `/cleanup-cards`."
- **Quality issues**: "5 cards too long. Consider splitting."

## Next Steps

Based on statistics:
- **Low coverage?** -> `/find-gaps`
- **Pending cards?** -> `/sync-cards`
- **Orphans found?** -> `/cleanup-cards`
- **Quality issues?** -> `/review-cards`

## Related

- `/find-gaps` - Detailed coverage gaps
- `/review-cards` - Card health check
- `/detect-changes` - Find modified notes
- [cli-reference.md](../_shared/cli-reference.md) - CLI commands

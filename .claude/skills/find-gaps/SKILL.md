---
name: find-gaps
description: Finds notes without Anki cards and identifies coverage gaps. Use to discover uncovered topics before bulk card generation or to check vault coverage.
allowed-tools: Read, Glob, Grep, Bash
---

# Find Coverage Gaps

Search for topics in Obsidian vault that don't have corresponding Anki cards.

## When to Use

- User asks to "find uncovered topics" or "check coverage"
- User wants to know "which notes need cards"
- User mentions "gaps" or "missing cards"
- Before bulk card generation

## Workflow

### Step 1: Determine Scope

- **By topic**: `/find-gaps --topic kotlin`
- **By directory**: `/find-gaps ~/path/to/notes`
- **All topics**: `/find-gaps --all`

### Step 2: Scan Notes

```bash
# Discover notes in vault
uv run anki-atlas obsidian-sync /path/to/vault --dry-run

# Check coverage for a topic
uv run anki-atlas coverage "programming/kotlin"

# Detect gaps
uv run anki-atlas gaps "programming" --min-coverage 1
```

### Step 3: Semantic Coverage (Optional)

```bash
uv run anki-atlas search "coroutines" --semantic --top 10
```

Returns topics with low semantic similarity to existing cards.

### Step 4: Present Gap Report

```
Coverage Gap Report
===================

Scope: Kotlin (45 notes)

Summary:
- Full coverage (EN + RU): 35 notes (78%)
- Partial: 5 notes (11%)
- None: 5 notes (11%)

PARTIAL COVERAGE:
1. q-coroutines-launch.md - Has: EN, Missing: RU
2. q-flow-operators.md - Has: RU, Missing: EN

NO COVERAGE:
1. q-channels-actors.md (hard)
2. q-sealed-classes.md (medium)
3. q-delegation-pattern.md (medium)

Priority Suggestions:
- 3 medium notes for quick wins
- 2 hard notes need careful crafting

Next Steps:
1. /generate-cards for specific notes
2. /bulk-process for batch generation
```

## Semantic vs Registry Gaps

| Approach | Finds | Use When |
|----------|-------|----------|
| Registry | Notes without any cards | Concrete coverage |
| Semantic | Concepts not well covered | Conceptual gaps |

Combine both for comprehensive analysis.

## Next Steps

After finding gaps:
- **Create cards?** -> `/generate-cards`
- **Batch process?** -> `/bulk-process`
- **View stats?** -> `/show-stats`

## Related

- `/generate-cards` - Create cards for uncovered notes
- `/bulk-process` - Batch process notes
- `/show-stats` - Coverage statistics
- [cli-reference.md](../_shared/cli-reference.md) - CLI commands

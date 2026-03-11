---
name: detect-changes
description: >
  Detects note modifications since last sync by comparing content hashes. Use to find stale cards, modified notes, or orphans before syncing.
  Don't use when the user wants to sync cards (use sync-cards) or delete orphans (use cleanup-cards).
argument-hint: "[path|--topic name]"
allowed-tools: Read, Glob, Grep, Bash
---

# Detect Vault Changes

Detect changes that require card updates by comparing note content with registry.

## When to Use

- User asks "what changed since last sync?"
- User mentions "updated notes" or "modified content"
- User wants to find "stale cards" or "outdated cards"
- Before syncing to check what needs updating

## Workflow

### Step 1: Determine Scope

- **By topic**: `/detect-changes --topic kotlin`
- **By directory**: `/detect-changes ~/path/to/notes`
- **Specific note**: `/detect-changes /path/to/note.md`

### Step 2: Scan Vault for Changes

```bash
uv run anki-atlas obsidian-sync /path/to/vault --dry-run
```

### Step 3: Check for Orphans

Search for cards whose source notes may have been deleted:

```bash
uv run anki-atlas search "topic" --semantic --top 10
```

### Step 4: Present Change Report

```
Change Detection Report
=======================

Scope: Kotlin (45 notes, 90 cards)

MODIFIED (content changed):
1. q-coroutines-launch.md
   - q-coroutines-launch-0-en [EN]
   - q-coroutines-launch-0-ru [RU]
   Last synced: 2024-01-15

2. q-flow-operators.md
   - Content updated (answer expanded)

Total: 3 cards (2 notes)

ORPHANED (source deleted):
1. q-removed-topic-0-en (Anki ID: 123)
2. q-removed-topic-0-ru (Anki ID: 124)

Total: 2 cards

UNCHANGED: 85 cards

Recommended Actions:
1. Re-generate 3 modified cards
2. Delete 2 orphan cards
3. Or sync all: /sync-cards
```

## Detection Categories

| Category | Meaning | Action |
|----------|---------|--------|
| MODIFIED | Content hash differs | Regenerate or force sync |
| ORPHAN | Source note deleted | Delete from Anki |
| UNCHANGED | Hash matches | None |

See [thresholds.md](../_shared/thresholds.md) for hash-based detection.

## Next Steps

Based on detection:
- **Modified cards?** -> `/generate-cards` or `/sync-cards`
- **Orphan cards?** -> `/cleanup-cards`
- **Check full status?** -> `/review-cards`

## Related

- `/cleanup-cards` - Delete orphan cards
- `/review-cards` - Compare with live Anki
- `/sync-cards` - Sync changes
- [thresholds.md](../_shared/thresholds.md) - Detection categories

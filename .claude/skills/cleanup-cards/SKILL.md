---
name: cleanup-cards
description: Cleans up orphaned, duplicate, or problematic cards from registry and Anki. Provides dry-run preview. Requires AnkiConnect MCP server for Anki operations.
allowed-tools: Read, Glob, Grep, Bash, mcp__anki__findNotes, mcp__anki__notesInfo, mcp__anki__deleteNotes
---

# Cleanup Cards

Clean up orphaned, duplicate, or problematic cards from registry and Anki.

## Prerequisites

- **Anki running** with AnkiConnect for Anki operations
- **MCP server** `@ankimcp/anki-mcp-server` for direct Anki access

## When to Use

- User asks to "clean up cards" or "remove orphans"
- User mentions "duplicate cards" or "stale cards"
- After `/detect-changes` identifies orphan cards
- Regular maintenance

## Workflow

### Step 1: Identify Issues

**Orphan Cards** (source notes deleted):
```bash
uv run anki-atlas search "orphan topic" --semantic --top 10
```

**Duplicate Cards** (same content hash):
```bash
uv run anki-atlas duplicates --threshold 0.95
```

See [thresholds.md](../_shared/thresholds.md) for threshold interpretation.

**Quality Issues**:
```bash
uv run anki-atlas validate card.md --quality
```

### Step 2: Present Cleanup Report

```
Cleanup Report [DRY RUN]
========================

ORPHAN CARDS (source deleted): 2
1. q-removed-topic-0-en (Anki ID: 123)
2. q-removed-topic-0-ru (Anki ID: 124)
   Action: Delete from registry and Anki

DUPLICATE CARDS (same content): 1
1. q-copy-0-en (duplicate of q-original-0-en)
   Action: Keep first, delete duplicate

QUALITY ISSUES: 2
1. q-verbose-0-en - Answer too long (1,250 chars)
2. q-no-tags-0-ru - Missing tags

Options:
[1] Delete orphans only
[2] Delete orphans + duplicates
[3] Export for manual review
```

### Step 3: Execute Cleanup

**Delete from Anki via MCP:**
```
mcp__anki__deleteNotes(notes=[123, 124], confirmDeletion=true)
```

### Step 4: Verify

```bash
uv run anki-atlas search "deleted topic" --semantic --top 5
```

## Safety Features

1. **Always dry-run first** - Show what would be deleted
2. **Require confirmation** - Never auto-delete
3. **Backup before delete** - Registry backup for destructive ops
4. **Verify after cleanup** - Confirm expected state

## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| Anki not running | Can't delete from Anki | Delete from registry only |
| Card not in Anki | Already deleted | Remove from registry |
| Permission denied | Registry locked | Retry |

## Next Steps

After cleanup:
- **Verify results?** -> `/show-stats`
- **Sync new cards?** -> `/sync-cards`
- **Find gaps?** -> `/find-gaps`

## Related

- `/detect-changes` - Find orphan cards
- `/review-cards` - Compare registry with Anki
- [thresholds.md](../_shared/thresholds.md) - Duplicate detection

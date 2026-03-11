---
name: review-cards
description: >
  Reviews and compares registry cards with live Anki state via MCP. Use for finding outdated cards, detecting discrepancies, or maintenance. Requires AnkiConnect MCP server.
  Don't use when the user wants statistics (use show-stats) or note quality analysis (use analyze-note).
allowed-tools: Read, Glob, Grep, Bash, mcp__anki__findNotes, mcp__anki__notesInfo, mcp__anki__deleteNotes, mcp__anki__updateNoteFields, mcp__anki__deckActions
---

# Review and Update Cards

Review existing cards using MCP tools for direct Anki access. Compare local registry with live Anki state.

## Prerequisites

- **Anki running** with AnkiConnect plugin
- **MCP server** `@ankimcp/anki-mcp-server` connected

## When to Use

- User asks to "review cards" or "check for outdated cards"
- User mentions "card maintenance" or "sync status"
- User wants to "find orphan cards"
- After note modifications

## Workflow

### Step 1: Query Local Index

```bash
uv run anki-atlas search "topic" --semantic --top 20 --verbose
```

### Step 2: Query Anki via MCP

```
mcp__anki__deckActions(action="listDecks")
mcp__anki__findNotes(query="deck:Kotlin Slug:_*")
mcp__anki__notesInfo(notes=[id1, id2, ...])  # Batch by 50
```

### Step 3: Compare and Categorize

| Category | Detection | Action |
|----------|-----------|--------|
| SYNCED | slug + hash match | None |
| LOCAL_ONLY | in local, not Anki | Sync via `/sync-cards` |
| ANKI_ONLY | in Anki, not local | Import or delete |
| HASH_MISMATCH | same slug, different hash | Update Anki |
| ORPHAN | source_path deleted | Delete from Anki |

See [thresholds.md](../_shared/thresholds.md) for detection details.

### Step 4: Present Report

```
Card Review Report (via MCP)
============================

Scope: deck:Kotlin (90 cards)

SYNCED: 76 cards (84%)
LOCAL_ONLY: 4 cards (4%) - need sync
HASH_MISMATCH: 6 cards (7%) - content differs
ORPHAN: 4 cards (4%) - source deleted

Recommended Actions:
1. Sync 4 LOCAL_ONLY cards
2. Resolve 6 HASH_MISMATCH cards
3. Delete 4 ORPHAN cards
```

### Step 5: Execute Actions

**Delete orphans:**
```
mcp__anki__deleteNotes(notes=[id1, id2], confirmDeletion=true)
```

**Update content:**
```
mcp__anki__updateNoteFields(note={
  "id": 123,
  "fields": {"Front": "...", "Back": "...", "ContentHash": "..."}
})
```

## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| Connection refused | Anki not running | Start Anki |
| MCP server not found | Not connected | Check config, restart Claude Code |
| Timeout | Large deck | Batch queries, use deck filter |

## Next Steps

After review:
- **Cards need sync?** -> `/sync-cards`
- **Orphans to delete?** -> `/cleanup-cards`
- **Check coverage?** -> `/find-gaps`
- **View statistics?** -> `/show-stats`

## Related

- `/cleanup-cards` - Remove orphan/duplicate cards
- `/detect-changes` - Find modified notes
- [thresholds.md](../_shared/thresholds.md) - Detection categories

---
name: bulk-process
description: Processes multiple Obsidian notes for batch card generation and sync. Use for folders, topics, or multiple notes at once. Tracks progress and provides summaries.
allowed-tools: Read, Glob, Grep, Bash
---

# Bulk Process Notes

Process multiple notes at once for batch card generation and synchronization.

**Card Model**: See [card-model.md](../_shared/card-model.md). Cards = topics x 2 (EN + RU).

## When to Use

- User mentions "bulk", "batch", or "multiple notes"
- User wants to "process a folder" or "sync all notes"
- User provides a directory path
- User wants to process existing notes by topic

## Bulk Processing Checklist

```
Bulk Processing Checklist:
- [ ] Discover notes in scope
- [ ] Check existing cards for each
- [ ] Present processing plan
- [ ] Get user approval
- [ ] Process notes sequentially
- [ ] Sync to Anki
- [ ] Update vector index
- [ ] Generate summary report
```

## Workflow

### Step 1: Discover Notes

```bash
uv run anki-atlas obsidian-sync /path/to/vault --dry-run
```

Or manually find markdown files. Exclude templates (`_template.md`) and drafts.

### Step 2: Analyze Each Note

```bash
uv run anki-atlas search "topic from note" --semantic --top 5
```

### Step 3: Present Processing Plan

```
Bulk Processing Plan
====================

Target: notes/python/ (18 notes)

Ready for cards (8 notes):
1. decorators.md - ~5 topics, ~10 cards
2. generators.md - ~4 topics, ~8 cards
...

Already synced (7 notes): [list]
Needs review (3 notes): [list]

Estimated: ~60 new cards (30 topics x 2 languages)

Options:
[1] Process all 8 ready notes
[2] Process specific notes
[3] Include "needs review"
```

### Step 4: Process Notes

For each note:
1. Read content
2. Identify card-worthy topics
3. Create cards manually
4. Track progress: `[3/8] Processing...`

### Step 5: Batch Sync

```bash
uv run anki-atlas obsidian-sync /path/to/vault
```

See [deck-naming.md](../_shared/deck-naming.md) for allowed decks.

### Step 6: Update Vector Index

**Required after bulk operations**:
```bash
uv run anki-atlas index
```

### Step 7: Generate Summary

```
Bulk Processing Complete
========================

Notes: 7/8 processed (1 error)
Cards: 56 created (28 topics x 2 languages)
  - English: 28 cards
  - Russian: 28 cards

By Note:
- decorators.md: 10 cards
- generators.md: 8 cards
...

Error: context-managers.md - Failed to parse Q&A

Next Steps:
1. Fix context-managers.md and retry
2. Review "needs review" notes
```

## Quality Validation

```bash
# Validate card content
uv run anki-atlas validate card.md --quality
```

## Error Handling

| Error | Cause | Recovery |
|-------|-------|----------|
| Parse failure | Invalid note structure | Skip, report at end |
| Sync timeout | Anki not responding | Retry or save for later |
| Duplicate slug | Card exists | Auto-resolve or prompt |

## Next Steps

After bulk processing:
- **Verify results?** -> `/show-stats`
- **Check coverage?** -> `/find-gaps`
- **Clean up?** -> `/cleanup-cards`
- **Review cards?** -> `/review-cards`

## Related

- `/generate-cards` - Single note processing
- `/sync-cards` - Sync operations
- `/find-gaps` - Find notes without cards first
- [cli-reference.md](../_shared/cli-reference.md) - CLI commands

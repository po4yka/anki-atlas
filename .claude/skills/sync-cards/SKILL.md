---
name: sync-cards
description: Syncs flashcards to Anki via AnkiConnect API. Use after generating cards, when pushing to Anki, or updating existing cards. Requires Anki running with AnkiConnect.
allowed-tools: Read, Bash
---

# Sync Cards to Anki

Sync cards to Anki via AnkiConnect API with full mapping system integration.

**Card Model**: See [card-model.md](../_shared/card-model.md).
**Deck Naming**: See [deck-naming.md](../_shared/deck-naming.md) for allowed decks.

## Prerequisites

1. **Cards registered** in the system (created via `/generate-cards`)
2. **Anki running** with AnkiConnect plugin (Code: 2055492159)
3. User has approved the cards

## When to Use

- After creating cards with `/generate-cards`
- User asks to "sync to Anki" or "push cards"
- User approves card proposals and wants them saved
- User wants to update existing cards

## Sync Validation Loop

Follow validate-fix-repeat pattern:

```
1. Dry-run validation
   |
2. Fix any issues
   |
3. Re-validate
   |
4. Sync when clean
```

## Workflow

### Step 1: Validate Cards

```bash
uv run anki-atlas validate card.md --quality
```

### Step 2: Preview Changes (Dry-Run)

```bash
uv run anki-atlas obsidian-sync /path/to/vault --dry-run
```

Output shows:
- **CREATE**: New cards
- **UPDATE**: Cards with changes
- **DELETE**: Cards to remove
- **UNCHANGED**: No changes

### Step 3: Sync Cards

```bash
uv run anki-atlas obsidian-sync /path/to/vault
```

### Step 4: Update Vector Index (Bulk Only)

For bulk sync (directories or multiple notes):

```bash
uv run anki-atlas index
```

### Step 5: Verify

```bash
uv run anki-atlas search "synced topic" --semantic --top 5
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Cannot connect" | Anki not running | Start Anki, verify AnkiConnect |
| "Duplicate card" | Exists in Anki | Check existing cards first |
| EN contains RU | Extraction issue | Fix regex boundaries |

## Next Steps

After syncing:
- **Verify sync worked?** -> `/show-stats`
- **Check for outdated cards?** -> `/review-cards`
- **Find more notes to process?** -> `/find-gaps`
- **Clean up orphans?** -> `/cleanup-cards`

## Related

- `/generate-cards` - Create cards before syncing
- `/bulk-process` - Sync multiple notes
- [cli-reference.md](../_shared/cli-reference.md) - CLI commands
- [deck-naming.md](../_shared/deck-naming.md) - Allowed decks

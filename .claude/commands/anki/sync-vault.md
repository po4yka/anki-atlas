---
description: Sync notes from Obsidian vault to Anki
argument-hint: "[vault-path] [--deck DeckName] [--tag filter-tag] [--dry-run]"
allowed-tools: [Read, Glob, Grep, Bash, mcp__anki__addNote, mcp__anki__findNotes, mcp__anki__notesInfo, mcp__anki__updateNoteFields, mcp__anki__deckActions, mcp__anki__modelNames]
---

# Sync Obsidian Vault to Anki

## Task

Sync flashcard-formatted notes from an Obsidian vault to Anki.

## Arguments

- `vault-path` (optional): Path to Obsidian vault (defaults to current directory)
- `--deck`: Target deck for new cards (default: "Obsidian")
- `--tag`: Only sync notes with this Obsidian tag
- `--dry-run`: Show what would be synced without making changes

## Expected Note Format

Obsidian notes should contain flashcard blocks:

```markdown
## Flashcards

Q: What is the question?
A: This is the answer.

Q: Another question?
A: Another answer.
```

Or cloze format:

```markdown
## Flashcards

The {{capital}} of France is {{Paris}}.
```

## Process

### 1. Validate Vault Path

- Check path exists
- Look for `.obsidian` directory to confirm it's a vault

### 2. Find Candidate Notes

```bash
# Find markdown files
fd '\.md$' [vault-path]
```

If `--tag` specified:
```
Grep for tag in frontmatter or content
```

### 3. Parse Flashcard Blocks

For each markdown file:
- Look for `## Flashcards` or `#flashcard` tagged sections
- Extract Q/A pairs or cloze deletions
- Preserve source file reference

### 4. Check Existing Cards

```
mcp__anki__findNotes with query for source file
```

- Track which cards already exist
- Identify cards needing updates vs new cards

### 5. Ensure Deck Exists

```
mcp__anki__deckActions with listDecks
```

If deck doesn't exist:
```
mcp__anki__deckActions with createDeck
```

### 6. Dry Run Report (if --dry-run)

Display:
- Files to process
- New cards to create
- Cards to update
- Cards to skip (duplicates)

Stop here if dry-run.

### 7. Sync Cards

For each new card:
```
mcp__anki__addNote
```

For each updated card:
```
mcp__anki__updateNoteFields
```

Tag all synced cards with:
- `obsidian-sync`
- Source file name

### 8. Report Results

Display summary:
- Cards created: N
- Cards updated: N
- Cards skipped: N
- Errors: N

List any errors with file names.

## Error Handling

| Error | Action |
|-------|--------|
| Vault not found | Ask for correct path |
| No flashcard blocks | List files checked, suggest format |
| Parse error | Show problematic content, skip file |
| Anki connection failed | Remind to start Anki |

## Sync Tracking

To enable update detection, cards include:
- Tag: `obsidian::filename` (sanitized)
- Field or tag with source hash

On re-sync:
- Find cards with matching source tag
- Compare content hash
- Update if changed

## Examples

```
/anki/sync-vault ~/Documents/Notes
/anki/sync-vault . --deck "Study Notes" --tag study
/anki/sync-vault ~/vault --dry-run
```

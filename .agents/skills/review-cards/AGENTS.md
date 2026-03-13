# review-cards - Quick Reference

Compare registry cards with live Anki via MCP. Requires Anki + AnkiConnect running.

## Workflow

1. Query local: `cargo run --bin anki-atlas -- search "topic" --semantic --top 20 --verbose`
2. Query Anki: `mcp__anki__findNotes`, `mcp__anki__notesInfo`
3. Categorize: SYNCED (match) | LOCAL_ONLY (sync) | ANKI_ONLY (import/delete) | HASH_MISMATCH (update) | ORPHAN (delete)
4. Present report, get confirmation, execute

## Commands

```bash
cargo run --bin anki-atlas -- search "topic" --semantic --top 20 --verbose
```

MCP: `mcp__anki__findNotes(query="deck:Kotlin")`, `deleteNotes`, `updateNoteFields`

## Refs

`docs/anki/thresholds.md`

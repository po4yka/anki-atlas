# sync-cards - Quick Reference

Sync flashcards to Anki via AnkiConnect. Validate-fix-repeat pattern.

## Prerequisites

Anki running with AnkiConnect (Code: 2055492159), cards approved by user.

## Workflow

1. Validate: `cargo run --bin anki-atlas -- validate card.md --quality`
2. Dry-run: `cargo run --bin anki-atlas -- obsidian-sync /path/to/vault --dry-run`
3. Fix issues, re-validate
4. Sync: `cargo run --bin anki-atlas -- obsidian-sync /path/to/vault`
5. Index (bulk): `cargo run --bin anki-atlas -- index`
6. Verify: `cargo run --bin anki-atlas -- search "topic" --semantic --top 5`

## Refs

`docs/anki/deck-naming.md`, `docs/anki/cli-reference.md`

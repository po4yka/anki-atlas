# bulk-process - Quick Reference

Batch process multiple notes for card generation and sync.

## Workflow

1. Discover: `cargo run --bin anki-atlas -- obsidian-sync /path/to/vault --dry-run`
2. Analyze each, present plan (ready/synced/needs-review, estimated cards)
3. Get approval, process sequentially, track `[N/M]`
4. Sync: `cargo run --bin anki-atlas -- obsidian-sync /path/to/vault`
5. Update index: `cargo run --bin anki-atlas -- index` (required after bulk)
6. Summary: notes processed, cards created, errors

## Refs

`docs/anki/card-model.md`, `docs/anki/deck-naming.md`

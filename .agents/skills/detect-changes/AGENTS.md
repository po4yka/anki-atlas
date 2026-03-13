# detect-changes - Quick Reference

Detect note modifications since last sync via content hash comparison.

## Workflow

1. Scope: by topic, directory, or note
2. Scan: `cargo run --bin anki-atlas -- obsidian-sync /path/to/vault --dry-run`
3. Categorize: MODIFIED (hash differs, regenerate) | ORPHAN (source deleted, delete) | UNCHANGED
4. Present change report with actions

## Refs

`docs/anki/thresholds.md`

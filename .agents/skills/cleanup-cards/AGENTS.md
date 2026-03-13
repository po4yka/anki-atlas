# cleanup-cards - Quick Reference

Remove orphaned, duplicate, or problematic cards. Always dry-run first, never auto-delete.

## Workflow

1. Find orphans (source deleted) and duplicates: `cargo run --bin anki-atlas -- duplicates --threshold 0.95`
2. Present dry-run report
3. Get user confirmation
4. Delete via MCP: `mcp__anki__deleteNotes(notes=[...], confirmDeletion=true)`
5. Verify deletion

## Refs

`docs/anki/thresholds.md`

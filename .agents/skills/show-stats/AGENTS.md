# show-stats - Quick Reference

Generate reports on sync status, card coverage, deck health.

## Commands

```bash
cargo run --bin anki-atlas -- coverage "programming"
cargo run --bin anki-atlas -- gaps "programming" --min-coverage 1
cargo run --bin anki-atlas -- duplicates --threshold 0.92
cargo run --bin anki-atlas -- search "test query" --verbose
```

Report types: Coverage, Gaps, Duplicates, Search health. Present summary with deck breakdown and actionable next steps.

## Refs

`docs/anki/cli-reference.md`

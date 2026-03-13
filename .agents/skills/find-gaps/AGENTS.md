# find-gaps - Quick Reference

Find notes without Anki cards and identify coverage gaps.

## Workflow

1. Scope: by topic, directory, or all
2. Scan: `cargo run --bin anki-atlas -- obsidian-sync /path/to/vault --dry-run`
3. Coverage: `cargo run --bin anki-atlas -- coverage "programming/kotlin"`
4. Gaps: `cargo run --bin anki-atlas -- gaps "programming" --min-coverage 1`
5. Report: full/partial/no coverage, priority suggestions

Registry gaps = notes without cards. Semantic gaps = concepts not well covered. Combine both.

## Refs

`docs/anki/cli-reference.md`

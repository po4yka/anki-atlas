# generate-cards - Quick Reference

Create bilingual (EN+RU) Anki flashcards. Read material, craft each card manually.

## Workflow

1. Read note, identify 3-10 key concepts
2. Search existing: `cargo run --bin anki-atlas -- search "topic" --semantic --top 10`
3. Check duplicates per card (see `docs/anki/thresholds.md`)
4. Craft EN card, then RU (Cyrillic only). Slug: `{note_id}-{index}-{lang}`
5. Present for approval, then sync

## Card Types

Definition (Basic Q&A), Syntax (Cloze `{{c1::...}}`), Comparison (X vs Y), Application (When to use)

## Commands

```bash
cargo run --bin anki-atlas -- search "topic" --semantic --top 10
cargo run --bin anki-atlas -- validate card.md --quality
cargo run --bin anki-atlas -- obsidian-sync /path/to/vault
```

## Refs

`docs/anki/card-model.md`, `docs/anki/tag-taxonomy.md`, `docs/anki/thresholds.md`

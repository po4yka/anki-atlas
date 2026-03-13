# analyze-note - Quick Reference

Analyze note structure and assess card-worthiness before flashcard creation.

## Workflow

1. Load: `cargo run --bin anki-atlas -- generate path/to/note.md --dry-run`
2. Score quality (1-10): structure, content depth, code examples
3. Assess load: LOW (1 card) | MEDIUM (2-3) | HIGH (4+, split)
4. Identify 3-10 topics: HIGH (definitions, distinctions) | MEDIUM (examples) | SKIP (obvious)
5. Check existing: `cargo run --bin anki-atlas -- search "topic" --semantic --top 10`
6. Report: score, topics, cards needed (topics x 2 for EN+RU)

## Quality: 9-10 ready | 7-8 minor fixes | 5-6 improve first | 1-4 major revision

## Refs

`docs/anki/card-model.md`

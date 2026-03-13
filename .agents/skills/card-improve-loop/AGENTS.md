# card-improve-loop - Quick Reference

Iterative card quality improvement. One card per iteration, validate after each change.

## Selection

```bash
anki-atlas weak-notes [--topic "kotlin"]
anki-atlas gaps "programming" --min-coverage 1
anki-atlas duplicates --threshold 0.92
```

## Cycle

1. Select one card (not in progress tracker)
2. Analyze quality (tags, content, anti-patterns)
3. Improve (rewrite, fix tags, split)
4. Validate: `anki-atlas validate card.md --quality`
5. Log to `.ralph/card-progress.md`
6. Commit

## Guardrails

One card/iteration. Re-read tracker each time. Max 2 retries before skip.

## Refs

`docs/anki/card-model.md`, `docs/anki/tag-taxonomy.md`, `docs/anki/thresholds.md`

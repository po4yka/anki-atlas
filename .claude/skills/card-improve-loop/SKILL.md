---
name: card-improve-loop
description: >
  Teaches ralph agents the anki-atlas CLI commands and workflows for reviewing
  and improving existing Anki cards one at a time. Used by the card-improve
  ralph loop preset.
allowed-tools: Read, Glob, Grep, Bash, Edit
---

# Card Improvement Loop -- Agent Reference

This skill is injected into ralph agents running the card-improve loop.
All content is self-contained (ralph cannot follow relative links).

## CLI Invocation

Use `cargo run --bin anki-atlas --` (or `anki-atlas` if installed).

## Card Selection Commands

```bash
# Find weak/low-quality notes
anki-atlas weak-notes                        # all
anki-atlas weak-notes --topic "kotlin"       # filtered by topic

# Find coverage gaps
anki-atlas gaps "programming"                # gaps in a topic tree
anki-atlas gaps "programming" --min-coverage 1

# Find near-duplicate notes
anki-atlas duplicates --threshold 0.92
anki-atlas duplicates --deck "Kotlin" --verbose

# Show topic coverage
anki-atlas coverage "programming/kotlin"
```

## Analysis Commands

```bash
# Validate card quality
anki-atlas validate card.md
anki-atlas validate card.md --quality

# Semantic duplicate search
anki-atlas search "question text" --semantic --top 5

# Tag audit
anki-atlas tag-audit tags.txt
anki-atlas tag-audit tags.txt --fix

# Topic taxonomy
anki-atlas topics
```

## Card Model

Every note produces at least 2 cards:
- English (`-en` suffix)
- Russian (`-ru` suffix, Cyrillic only -- never transliteration)

**Slug format**: `{note_id}-{index}-{language}` (e.g., `q-coroutines-0-en`)

### Card Principles

1. **Atomic**: one fact per card
2. **Active recall**: question requires thinking, not recognition
3. **Concise**: answers under 100 words
4. **Context**: card makes sense in isolation
5. **Formatted**: bold terms, code blocks, lists

### Anti-Patterns

| Bad Pattern | Fix |
|-------------|-----|
| "List all X" (set) | Split into N separate cards |
| "Name sequence" (enum) | Overlapping cloze |
| Yes/no question | Ask "how/why" instead |
| "Explain X" (vague) | Split: definition + mechanism + comparison |
| Similar cards confuse | Add prefix: `[StateFlow]`, `[SharedFlow]` |

### Question Type Distribution

| Type | Target % | Use For |
|------|----------|---------|
| Definition | 40% | Core concepts |
| Comparison | 25% | X vs Y distinctions |
| Application | 20% | "When to use..." |
| Cloze | 15% | Syntax, formulas |

## Tag Taxonomy

### Required Tags

Every card MUST have:
1. One difficulty tag: `difficulty::easy`, `difficulty::medium`, `difficulty::hard`
2. One or more topic tags from the taxonomy

### Format Rules

| Rule | Convention | Example |
|------|-----------|---------|
| Hierarchy separator | `::` | `kotlin::coroutines` |
| Word separator | `-` (kebab-case) | `state-management` |
| Code identifiers | Original casing | `ArrayList` |
| Max depth | 2 levels | `kotlin::coroutines` |
| Case | lowercase (except code IDs) | `cs::algorithms` |

### Domain Prefixes

| Prefix | Scope |
|--------|-------|
| `android::` | Android framework, Jetpack |
| `kotlin::` | Kotlin language features |
| `cs::` | Computer science fundamentals |
| `topic::` | Cross-cutting themes |
| `difficulty::` | Card difficulty level |
| `lang::` | Natural/programming language |
| `source::` | Card origin tracking |
| `context::` | Study context |

## Similarity Thresholds

| Score | Classification | Action |
|-------|----------------|--------|
| > 0.95 | Exact duplicate | Delete or merge |
| 0.85-0.95 | Near duplicate | Add `[context]` prefix to distinguish |
| 0.70-0.85 | Related content | Consider comparison card |
| < 0.70 | Distinct | No action needed |

## Progress Tracking

File: `.ralph/card-progress.md`

Format (markdown table):

```markdown
| Card ID | Timestamp | Action | Quality Before | Quality After |
|---------|-----------|--------|----------------|---------------|
| q-coroutines-0-en | 2026-03-11T10:00 | fixed-tags | fail | pass |
| q-flow-1-ru | 2026-03-11T10:05 | rewritten | weak | pass |
| q-channels-2-en | 2026-03-11T10:10 | skipped | fail (2 retries) | - |
```

### Rules

- Re-read `.ralph/card-progress.md` at the start of every iteration
- Never re-process a card already listed
- One card per iteration
- If validation fails twice on the same card, skip and log reason
- Git commit after each successful improvement

## Guardrails

1. Fresh context each iteration -- always re-read progress tracker
2. One card per iteration -- never batch
3. Validate after every change
4. Skip already-processed cards
5. Max 2 retries per card before skipping
6. Commit after each successful round

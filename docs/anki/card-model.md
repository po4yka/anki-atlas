# Card Model Reference

## Bilingual Requirement

Every note produces at least 2 cards:
- One English (`-en` suffix)
- One Russian (`-ru` suffix, Cyrillic only)

**Russian = Cyrillic ONLY**: Never use transliteration.

| WRONG | CORRECT |
|-------|---------|
| "V chem raznitsa mezhdu..." | "В чём разница между..." |

**Exception**: Code stays Latin (`associateBy`, `Map<Int, User>`)

**Valid languages**: `en`, `ru`, `de`, `fr`, `es`, `it`, `pt`, `zh`, `ja`, `ko`
(source: `crates/card/src/models.rs` `VALID_LANGUAGES`)

## Note vs Card

1 Anki **note** produces 1+ **cards**. APF::Simple creates 1 card per note.
Bilingual workflow = 2 separate notes (one EN, one RU) sharing a slug base:
`q-coroutines-0-en` + `q-coroutines-0-ru`.

## Note Types

| Note Type | ~Notes | Fields | Usage |
|-----------|--------|--------|-------|
| `APF::Simple` | 3305 | 13 (see below) | Primary — generated cards |
| `Basic+` | 412 | Front, Back | Legacy manual cards |
| `Basic` | 90 | Front, Back | Legacy manual cards |
| `APF::Cloze` | — | (system-defined) | Cloze deletions |
| `Cloze` | — | Text, Back Extra | Legacy cloze |

System-accepted types (`VALID_NOTE_TYPES`): `APF::Simple`, `APF::Cloze`, `Basic`, `Cloze`.

## APF::Simple Fields

| Field | Required | Description |
|-------|----------|-------------|
| Front | yes | Question text (HTML) |
| Back | yes | Answer text (HTML) |
| Slug | yes | Unique ID: `{note_id}-{index}-{lang}` |
| ContentHash | yes | Change detection hash (see below) |
| SourceLink | no | Link to source Obsidian note |
| Title | no | Card title for display |
| Subtitle | no | Subtitle / topic context |
| Syntax | no | Inline code syntax hint |
| Sample | no | Code block example |
| KeyPoint | no | Key point code block |
| KeyPointNotes | no | Explanation of key point |
| OtherNotes | no | Additional notes |
| Markdown | no | Raw markdown source |

## Card Slug Format

```
{note_id}-{index}-{language}
```

Examples:
- `q-coroutines-0-en` (English)
- `q-coroutines-0-ru` (Russian)

## Content Hash

Two hash variants exist:

- **Card-level** (6 hex chars): SHA-256[:6] of `"{apf_html}|{note_type}|{sorted_tags}"` — stored in the ContentHash field, used for HASH_MISMATCH detection.
- **Slug-level** (12 hex chars): SHA-256[:12] of `"{front}|{back}"` — used by `SlugService` for content-addressed dedup.

## Card Creation Principles

1. **Atomic**: One fact per card
2. **Active recall**: Question requires thinking, not recognition
3. **Concise**: Answers under 100 words
4. **Context**: Card makes sense in isolation
5. **Formatted**: Bold terms, code blocks, lists

## SuperMemo Anti-Patterns

| Bad Pattern | Fix |
|-------------|-----|
| "List all X" (set) | Split into N separate cards |
| "Name sequence" (enum) | Overlapping cloze |
| Yes/no question | Ask "how/why" instead |
| "Explain X" (vague) | Split: definition + mechanism + comparison |
| Similar cards confuse | Add prefix: `[StateFlow]`, `[SharedFlow]` |

## Question Type Distribution

| Type | Percentage | Use For |
|------|------------|---------|
| Definition | 40% | Core concepts, definitions |
| Comparison | 25% | X vs Y distinctions |
| Application | 20% | "When to use..." practical |
| Cloze | 15% | Syntax, formulas |

See `CLAUDE.md` for complete card creation guide.

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

## Card Slug Format

```
{note_id}-{index}-{language}
```

Examples:
- `q-coroutines-0-en` (English)
- `q-coroutines-0-ru` (Russian)

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

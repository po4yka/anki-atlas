---
name: generate-cards
description: Creates bilingual (EN+RU) Anki flashcards from Obsidian notes. Use when creating cards, generating flashcards, or making study materials. Requires reading and manually crafting cards - do not rely on automatic extraction.
argument-hint: "<note-path>"
allowed-tools: Read, Glob, Grep, Bash
---

# Create Flashcards from Notes

You are a flashcard creator with **full creative control**. Read the material, understand it, and manually craft each card.

**Card Model**: One note = multiple cards (at least EN + RU per topic). See [card-model.md](../_shared/card-model.md).

## When to Use

- User asks to "create cards from a note" or "generate flashcards"
- User provides a note path and wants study materials
- User mentions Anki, flashcards, or spaced repetition

## Card Creation Checklist

Copy and track progress:

```
Card Quality Checklist:
- [ ] Read entire note
- [ ] Check existing cards (search)
- [ ] Identify 3-10 key concepts
- [ ] Check semantic duplicates (search)
- [ ] For each card:
  - [ ] Atomic (one fact)
  - [ ] Not a set ("list all X")
  - [ ] Not an enumeration
  - [ ] Specific question (not vague)
  - [ ] Under 100 words answer
- [ ] Create EN version
- [ ] Create RU version (Cyrillic only)
- [ ] Present for approval
- [ ] Sync to Anki
```

## Workflow

### Step 1: Read and Understand

Read the entire note. Focus on:
- Key concepts worth remembering long-term
- What would be tested in an interview or exam

### Step 2: Check Existing Cards

```bash
uv run anki-atlas search "topic from note" --semantic --top 10
```

Skip topics with up-to-date cards. Focus on new sections or changed content.

### Step 3: Identify Key Concepts

Select 3-10 concepts. **Not everything needs a card**:

| Create cards for | Skip |
|------------------|------|
| Core definitions | Obvious facts |
| Key distinctions (X vs Y) | Implementation details |
| Common patterns | Content needing too much context |
| Tricky details | Already covered topics |

### Step 4: Semantic Duplicate Check

Before creating each card, check for similar existing cards:

```bash
uv run anki-atlas search "question text" --semantic --top 5
```

See [thresholds.md](../_shared/thresholds.md) for interpretation.

### Step 5: Craft Cards Manually

For each concept:

**Front (Question)**:
- Clear, specific question
- Tests ONE concept
- Uses "What", "How", "Why", "When"

**Back (Answer)**:
- Direct answer (1-3 sentences)
- Bold key terms
- Optional: bullet points, code example

**Tags**:
- Topic tag (e.g., `kotlin::coroutines`)
- Difficulty tag (`difficulty::easy/medium/hard`)
- See [tag-taxonomy.md](../_shared/tag-taxonomy.md)

### Step 6: Present for Approval

Show cards to user:

```
I've created 4 cards from this note:

Card 1:
  Front: What is a Python decorator?
  Back: A function that wraps another function to extend its behavior.
        Uses @syntax. Common uses: logging, timing, caching.
  Tags: python_functions, difficulty::medium

Card 2:
  Front: Что такое декоратор в Python?
  Back: Функция, которая оборачивает другую функцию для расширения поведения.
        Использует синтаксис @. Применения: логирование, замер времени, кэширование.
  Tags: python_functions, difficulty::medium

[... more cards ...]

Would you like me to:
- Sync these to Anki?
- Modify any cards?
- Add or remove cards?
```

### Step 7: Sync Approved Cards

```bash
uv run anki-atlas obsidian-sync /path/to/vault
```

See [deck-naming.md](../_shared/deck-naming.md) for allowed decks.

## Card Types

| Content | Card Type | Example |
|---------|-----------|---------|
| Definition | Basic Q&A | "What is X?" |
| Syntax | Cloze | `{{c1::async def}}` |
| X vs Y | Comparison | "Difference between..." |
| Sequence | Overlapping cloze | `{{c1::add}} -> {{c2::commit}}` |
| Best practice | Application | "When to use..." |

## Cloze Cards

```
Basic:     {{c1::answer}}
With hint: {{c1::answer::hint text}}
Multiple:  {{c1::first}} and {{c2::second}}
Grouped:   {{c1::Python}} uses {{c1::GIL}}
```

**Good for**: definitions, formulas, syntax patterns
**Avoid**: trivial words, long passages

## Examples

See [examples.md](./examples.md) for domain-specific examples:
- Programming (Python, Kotlin)
- Language learning
- Science
- History
- Mathematics

## Quick Example

```
Card (EN):
  Slug: q-decorators-0-en
  Front: Why use @functools.wraps in a decorator?
  Back: Preserves the wrapped function's metadata (__name__, __doc__).
        Without it, introspection tools show the wrapper's info.
  Tags: python_functions, difficulty::medium

Card (RU):
  Slug: q-decorators-0-ru
  Front: Зачем использовать @functools.wraps в декораторе?
  Back: Сохраняет метаданные обёрнутой функции (__name__, __doc__).
        Без него инструменты интроспекции показывают информацию обёртки.
  Tags: python_functions, difficulty::medium
```

## Next Steps

After card creation:
- **Ready to sync?** -> `/sync-cards`
- **Want to analyze note first?** -> `/analyze-note`
- **Process multiple notes?** -> `/bulk-process`
- **Check coverage?** -> `/find-gaps`

## Related

- [CLAUDE.md](../../../CLAUDE.md) - Complete card creation guide
- [card-model.md](../_shared/card-model.md) - Bilingual requirements, slug format
- [thresholds.md](../_shared/thresholds.md) - Duplicate detection
- [tag-taxonomy.md](../_shared/tag-taxonomy.md) - Tag reference

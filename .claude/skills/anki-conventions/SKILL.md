---
name: anki-conventions
user-invocable: false
description: >
  Anki card creation patterns, query syntax, FSRS settings, and spaced repetition principles.
  Use when creating flashcards, querying decks, discussing learning strategies, managing backlogs,
  or working with AnkiConnect API. Triggers on: anki, flashcard, spaced repetition, deck, card,
  note type, cloze, review, fsrs, retention, interval, backlog, leech, ease, programming card,
  code card, suspend, due, overdue, tag convention, tag format, tag prefix, tagging.
---

# Anki Conventions

## Card Philosophy: Mastery-Oriented

Cards are designed for deep understanding and terminology mastery, not elementary-level learning. Cards build the mental models AI cannot substitute.

| Principle | Description |
|-----------|-------------|
| **Precise terminology** | Use correct technical terms, not simplified language |
| **Depth over brevity** | Answers can include nuanced explanations and connections |
| **Conceptual connections** | Link to related concepts and underlying principles |
| **Expert-level questions** | Require understanding, not just recognition |
| **Why and how** | Focus on reasoning alongside facts |

### Elementary vs Mastery

| Aspect | Elementary (avoid) | Mastery (prefer) |
|--------|-------------------|------------------|
| Question style | "What does X do?" | "When would you choose X over Y, and why?" |
| Terminology | "The thing that stores data" | "The data structure with O(1) lookup" |
| Answer depth | "X is a type of Y" | "X is a Y that provides Z because of W" |
| Connections | Isolated fact | Links to related concepts |
| Target | Recognition | Application and reasoning |

### Mastery Quality Checklist

- [ ] Uses precise technical terminology
- [ ] Question requires understanding, not just recall
- [ ] Answer explains "why" or "how", not just "what"
- [ ] Connects to related concepts or principles
- [ ] Would help someone apply knowledge, not just recognize it
- [ ] Tests at the level of a practitioner, not a beginner

## User Preferences

**Longer technical cards are acceptable.** For programming topics, the atomic rule is relaxed:
- Code snippets need sufficient context
- Multi-line examples demonstrating patterns are fine
- Gotcha explanations can include related pitfalls

See `references/programming-cards.md` for technical card patterns.

## Quick Reference

### Query Syntax (Most Common)

| Query | Description |
|-------|-------------|
| `deck:Name` | Cards in deck |
| `deck:Parent::Child` | Cards in subdeck |
| `tag:tagname` | Cards with tag |
| `-tag:tagname` | Cards without tag |
| `is:due` | Due for review |
| `is:new` | Unstudied cards |
| `is:suspended` | Suspended cards |
| `prop:due<0` | Overdue cards |
| `added:N` | Added in last N days |
| `front:text` | Search Front field |

Combine with spaces (AND) or `OR`.

### Card Design Principles

1. **Atomic**: One concept per card (relaxed for programming)
2. **Precise**: Use correct terminology, not simplified language
3. **Deep**: Test understanding and reasoning, not just recall
4. **Connected**: Link to related concepts and principles
5. **Applicable**: Enable applying knowledge, not just recognizing it

### Tag Conventions

| Rule | Convention | Example |
|------|-----------|---------|
| Hierarchy | `::` separator | `android::compose` |
| Words | kebab-case | `state-management` |
| Code IDs | Original casing | `ArrayList`, `WorkManager` |
| Max depth | 2 levels | `prefix::topic` |
| Prefixes | `android::`, `kotlin::`, `cs::`, `topic::`, `difficulty::`, `lang::`, `source::`, `context::` | |

See `references/tag-conventions.md` for full rules, normalization, and validation checklist.

### FSRS Quick Setup

FSRS-7 is the **final major version**. Current: FSRS-6 (Anki 25.02+).

1. Enable FSRS in deck options
2. Set desired retention to **0.90**
3. Learning step: **15m or 30m** (single step, completable same day)
4. Click "Optimize" monthly
5. Use **Again** and **Good** primarily (never "Hard" when forgotten)

### Note Types

| Type | Use Case |
|------|----------|
| Basic | Simple Q&A |
| Basic (reversed) | Learn both directions |
| Cloze | Fill-in-the-blank, lists, definitions |

### MCP Tools Available

```
mcp__anki__addNote        # Create note
mcp__anki__findNotes      # Search notes
mcp__anki__notesInfo      # Get note details
mcp__anki__updateNoteFields  # Modify note
mcp__anki__deleteNotes    # Remove notes
mcp__anki__deckActions    # Deck operations
mcp__anki__modelNames     # List note types
mcp__anki__sync           # Sync with AnkiWeb
```

### Backlog Quick Recovery

```
# Find overdue cards
prop:due<0

# Suspend, tag, then unsuspend batches daily
```

### Verification Pattern

Always verify operations:
1. After `addNote`: Check returned ID is not null
2. After `updateNoteFields`: Re-fetch to confirm
3. After `deleteNotes`: Verify removal

## Detailed References

| Topic | Reference File |
|-------|----------------|
| Query syntax | `references/query-syntax.md` |
| Card patterns | `references/card-patterns.md` |
| Note types | `references/note-types.md` |
| **FSRS settings** | `references/fsrs-settings.md` |
| **Programming cards** | `references/programming-cards.md` |
| **Deck organization** | `references/deck-organization.md` |
| **Card maintenance** | `references/card-maintenance.md` |
| **Tag conventions** | `references/tag-conventions.md` |
| **Learning in AI age** | `references/learning-in-ai-age.md` |
| **Troubleshooting** | `references/troubleshooting.md` |

---
description: Review and improve an existing Anki card's quality
argument-hint: "<card-id | search-query> [--apply] [--deck DeckName]"
allowed-tools: [Read, mcp__anki__findNotes, mcp__anki__notesInfo, mcp__anki__updateNoteFields, mcp__anki__modelFieldNames, mcp__anki__deckActions, mcp__anki__tagActions, mcp__anki__getTags]
---

# Improve Anki Card

## Queue-Driven Mode (Preferred)

If the cardloop system is initialized (`.cardloop/` directory exists), prefer the queue-driven workflow:

1. Check the queue: `anki-atlas cardloop next -n 1`
2. If items exist, work on the next item instead of ad-hoc searching
3. After fixing, resolve: `anki-atlas cardloop resolve <id> --attest "..." --registry .cardloop/cards.db`

This ensures systematic coverage and prevents re-work on already-addressed cards.

## Task

Analyze an existing Anki card against best practices and suggest (or apply) improvements.

## Arguments

- `card-id`: Note ID to improve (numeric)
- `search-query`: Query to find card(s) - e.g., `"front:recursion"`, `"tag:leech"`
- `--apply`: Apply suggested improvements automatically
- `--deck DeckName`: Limit search to specific deck

## Process

### 1. Find the Card

If numeric ID provided:
```
mcp__anki__notesInfo with [id]
```

If search query provided:
```
mcp__anki__findNotes with query
```

If multiple results, show list and ask user to select.

### 2. Fetch Card Details

```
mcp__anki__notesInfo with selected note ID
mcp__anki__modelFieldNames to understand field structure
```

Display current card:
```
Note ID: [id]
Note Type: [type]
Deck: [deck]
Tags: [tags]

Front:
─────────────────────
[front content]

Back:
─────────────────────
[back content]
```

### 3. Analyze Against Best Practices

Evaluate the card against mastery-oriented criteria:

#### Mastery Depth Check
- [ ] Tests understanding, not just recall
- [ ] Asks "why" or "when", not just "what"
- [ ] Uses precise technical terminology
- [ ] Connects to related concepts or principles
- [ ] Would help apply knowledge, not just recognize it

#### Elementary Patterns to Improve
- [ ] "What is X?" -> "When would you choose X over Y?"
- [ ] "Define X" -> "How does X differ from Y?"
- [ ] Single-fact answer -> Include trade-offs or reasoning
- [ ] Isolated concept -> Connect to related principles

#### Atomic Check
- [ ] Tests one concept (with appropriate depth)
- [ ] Could be split if covering multiple unrelated concepts

#### Answer Quality
- [ ] Includes reasoning, not just facts
- [ ] For technical: includes trade-offs and when-to-use
- [ ] For code: includes why, not just syntax

#### Technical Card Rules (if programming-related)
- [ ] Tests reasoning about patterns, not just syntax
- [ ] Includes trade-offs for decisions
- [ ] Explains "why" alongside "what"
- [ ] Gotchas include prevention strategies

#### Common Issues
- [ ] Surface-level question - tests recognition, not understanding
- [ ] Missing reasoning - answer lacks "why" or "how"
- [ ] Simplified terminology - should use precise technical terms
- [ ] Isolated fact - needs connection to related concepts

#### Tag Convention Check
- [ ] Uses `::` for hierarchy (not `_`, `/`, or `.`)
- [ ] Uses `-` between words (not `_` or camelCase)
- [ ] Max 2 hierarchy levels
- [ ] Has domain prefix for categorizable concepts (`android::`, `kotlin::`, `cs::`, etc.)
- [ ] No status/process tags (use Anki flags instead)
- [ ] Code identifiers preserved in original casing

### 4. Generate Improvement Suggestions

Present findings:

```
## Analysis Results

### Issues Found

1. **[Issue type]**: [Description]
   - Current: [problematic part]
   - Problem: [why it's an issue]

### Suggested Improvements

**Original Front:**
[original]

**Improved Front:**
[improved version]

**Original Back:**
[original]

**Improved Back:**
[improved version]

### Tag Issues

| Current Tag | Issue | Suggested Fix |
|-------------|-------|---------------|
| [tag] | [underscore/slash/missing prefix/etc.] | [normalized tag] |

### Additional Recommendations

- [ ] [recommendation 1]
- [ ] [recommendation 2]
```

When `--apply` is used: fix non-canonical tags via `mcp__anki__tagActions` (replaceTags).

### 5. Apply Changes (if --apply or user confirms)

If improvements suggested and user wants to apply:

```
mcp__anki__updateNoteFields with:
- id: [note id]
- fields: { updated fields }
```

**Important:** Verify the note is not open in Anki's browser (updates silently fail if it is).

After update:
```
mcp__anki__notesInfo to verify changes applied
```

### 6. Summary

Display:
```
Card Improvement Complete
─────────────────────────
Note ID: [id]
Changes: [list of changes made]
Status: [Applied / Suggestions only]

Tip: Close and reopen Anki browser to see changes.
```

## Improvement Patterns

### Splitting Cards

If a card tests multiple facts, offer to create additional cards:

```
This card tests 3 concepts. Split into:

Card 1: [focused question 1]
Card 2: [focused question 2]
Card 3: [focused question 3]

Create additional cards? [y/N]
```

### Reformulating Questions (Elementary -> Mastery)

| Elementary Pattern | Mastery Pattern |
|-------------------|-----------------|
| "What is X?" | "When would you choose X over Y, and what trade-offs does this involve?" |
| "Define X" | "How does X differ from [similar concept], and when is each appropriate?" |
| "What does X do?" | "Why was X designed this way, and what problems does it solve?" |
| "Yes/No: Is X true?" | "Under what conditions is X true, and what are the implications?" |
| "List all X" | Multiple cards explaining why each item matters |
| "Explain X" | "What reasoning underlies X, and how would you apply it?" |

### Improving Code Cards (Mastery)

| Issue | Mastery Fix |
|-------|-------------|
| Tests syntax only | Add "when to use" and trade-offs |
| Missing "why" | Explain reasoning behind the pattern |
| No decision guidance | Add when to choose this vs alternatives |
| Isolated example | Connect to underlying principle |
| Too long (>10 lines) | Extract core pattern, keep reasoning |
| Missing context | Add language tag, use case, pitfalls |

## Error Handling

| Error | Action |
|-------|--------|
| Card not found | Show search tips, suggest queries |
| Multiple matches | List cards, ask user to select |
| Update failed | Check if card is open in browser |
| No improvements needed | Confirm card passes all checks |

## Examples

```
# Improve by ID
/anki/improve-card 1234567890

# Find and improve
/anki/improve-card "front:recursion"

# Improve leeches in a deck
/anki/improve-card "tag:leech" --deck Programming

# Auto-apply improvements
/anki/improve-card 1234567890 --apply
```

## User Preference Note

**Mastery-oriented cards preferred.** Cards should be designed for deep understanding:
- Use precise technical terminology, not simplified language
- Test understanding and reasoning, not just recall
- Include trade-offs, "why", and conceptual connections
- Longer answers with depth are better than brief surface-level facts

For programming/technical cards, longer answers with code snippets, trade-offs, and contextual explanations are expected. The atomic rule is relaxed for technical content that requires depth.

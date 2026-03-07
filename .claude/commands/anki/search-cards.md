---
description: Search and display Anki cards matching a query
argument-hint: "<query> [--limit N] [--show-answer] [--deck filter]"
allowed-tools: [mcp__anki__findNotes, mcp__anki__notesInfo, mcp__anki__deckActions]
---

# Search Anki Cards

## Task

Search for Anki cards using Anki's query syntax and display results.

## Arguments

- `query` (required): Anki search query
- `--limit N`: Maximum results to display (default: 10)
- `--show-answer`: Include answer/back content in results
- `--deck`: Filter to specific deck

## Query Syntax Quick Reference

| Query | Matches |
|-------|---------|
| `word` | Cards containing "word" |
| `"exact phrase"` | Exact phrase match |
| `deck:Name` | Cards in deck |
| `tag:tagname` | Cards with tag |
| `is:due` | Due for review |
| `is:new` | New cards |
| `added:7` | Added in last 7 days |
| `front:text` | Search front field |

Combine with spaces (AND) or `OR`.

## Process

### 1. Parse Query

If `--deck` provided, prepend to query:
```
deck:[deck-name] [original-query]
```

Validate query syntax.

### 2. Execute Search

```
mcp__anki__findNotes with query
```

### 3. Get Note Details

```
mcp__anki__notesInfo with found note IDs (limited)
```

### 4. Display Results

#### Default (questions only)

```
Search: "tag:python is:due"
Found: 45 cards (showing 10)
═══════════════════════════

1. [ID: 1234567890] deck:Programming
   Q: What is a list comprehension in Python?
   Tags: python, syntax

2. [ID: 1234567891] deck:Programming
   Q: How do you define a generator function?
   Tags: python, generators

3. [ID: 1234567892] deck:Programming
   Q: What does the @property decorator do?
   Tags: python, decorators

... (7 more)

Use --show-answer to see answers
Use --limit N to see more results
```

#### With Answers (--show-answer)

```
Search: "tag:python" --show-answer
Found: 45 cards (showing 10)
═══════════════════════════

1. [ID: 1234567890] deck:Programming
   Q: What is a list comprehension in Python?
   A: A concise way to create lists using a single line
      of code: [expr for item in iterable if condition]
   Tags: python, syntax
   ───────────────────

2. [ID: 1234567891] deck:Programming
   Q: How do you define a generator function?
   A: Use the 'yield' keyword instead of 'return'
   Tags: python, generators
   ───────────────────

...
```

### 5. Result Summary

```
───────────────────────────
Results: 10 of 45 total
States: 12 new, 28 due, 5 suspended

Actions:
- /anki/search-cards "tag:python" --limit 20  (show more)
- /anki/review-session Programming             (review these)
```

## Advanced Queries

### Find struggling cards

```
/anki/search-cards "prop:lapses>=3"
```

### Find cards added recently

```
/anki/search-cards "added:7 deck:MyDeck"
```

### Find untagged cards

```
/anki/search-cards "-tag:* deck:Default"
```

### Find cards by content

```
/anki/search-cards "front:*recursion*"
```

### Find suspended cards

```
/anki/search-cards "is:suspended deck:Programming"
```

## Error Handling

| Error | Action |
|-------|--------|
| Invalid query syntax | Show syntax help, suggest correction |
| No results | Suggest broader query or check spelling |
| Too many results | Recommend narrowing with deck/tag filters |
| Connection failed | Check Anki is running |

## Output Options

Results can be used for:
- Identifying cards to review
- Finding cards to edit
- Auditing deck contents
- Locating duplicates

## Examples

```
/anki/search-cards "is:due"
/anki/search-cards "deck:Spanish tag:verbs" --show-answer
/anki/search-cards "added:30" --limit 50
/anki/search-cards "front:capital" --deck Geography
/anki/search-cards "prop:ease<2" --show-answer
```

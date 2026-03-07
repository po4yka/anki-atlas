---
description: Start an interactive Anki review session
argument-hint: "[deck] [--limit N] [--new-only] [--due-only]"
allowed-tools: [mcp__anki__get_due_cards, mcp__anki__present_card, mcp__anki__rate_card, mcp__anki__deckActions, mcp__anki__findNotes, mcp__anki__notesInfo, mcp__anki__sync]
---

# Interactive Anki Review Session

## Task

Conduct an interactive flashcard review session within Claude Code.

## Arguments

- `deck` (optional): Deck to review (default: all due cards)
- `--limit N`: Maximum cards to review (default: 20)
- `--new-only`: Only show new/unstudied cards
- `--due-only`: Only show due cards (skip new)

## Process

### 1. Sync First (Optional)

Ask user if they want to sync with AnkiWeb first:

```
mcp__anki__sync
```

### 2. Get Review Queue

```
mcp__anki__get_due_cards
```

Or if deck specified:
```
mcp__anki__findNotes with query "deck:[name] is:due"
```

Filter based on flags:
- `--new-only`: `is:new`
- `--due-only`: `is:due -is:new`

### 3. Session Setup

Display:
- Deck being reviewed
- Total cards in queue
- New / Learning / Review breakdown
- Session limit

### 4. Review Loop

For each card (up to limit):

#### Present Card

```
mcp__anki__present_card
```

Display:
```
Card [N] of [Total]
─────────────────────
[Front content]

Press Enter to reveal answer...
```

Wait for user input.

#### Reveal Answer

Display:
```
[Front content]
─────────────────────
[Back content]

Rate your recall:
1. Again (forgot)
2. Hard (difficult)
3. Good (correct)
4. Easy (effortless)
```

#### Record Rating

```
mcp__anki__rate_card with selected ease
```

Ease mapping:
- 1 = Again
- 2 = Hard
- 3 = Good
- 4 = Easy

#### Continue or Stop

After each card:
- Show progress (X/Y completed)
- Offer to continue or end session

User can type:
- Number (1-4) to rate
- `s` to skip
- `q` to quit session
- `e` to edit card (opens in Anki)

### 5. Session Summary

At end of session, display:

```
Session Complete
────────────────
Cards reviewed: N
  Again: N
  Hard: N
  Good: N
  Easy: N

Average ease: X.X
Time spent: ~N minutes

Remaining due today: N
```

### 6. Sync Results (Optional)

Ask if user wants to sync results to AnkiWeb:

```
mcp__anki__sync
```

## Interaction Model

Since Claude Code is text-based, the review uses a conversational flow:

1. Show question, wait for "reveal" or Enter
2. Show answer with rating options
3. Accept rating (1-4) or command (s/q/e)
4. Repeat until limit or quit

## Error Handling

| Error | Action |
|-------|--------|
| No due cards | Report "Nothing to review!" with next due time |
| Deck not found | List available decks |
| Card display error | Skip card, log error |
| Rating failed | Retry once, then skip |

## Examples

```
/anki/review-session
/anki/review-session "Japanese Vocabulary" --limit 10
/anki/review-session --new-only --limit 5
/anki/review-session Programming --due-only
```

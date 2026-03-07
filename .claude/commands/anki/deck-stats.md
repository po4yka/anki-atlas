---
description: Display Anki deck statistics and study progress
argument-hint: "[deck-name] [--all] [--verbose]"
allowed-tools: [mcp__anki__deckActions, mcp__anki__findNotes, mcp__anki__notesInfo]
---

# Anki Deck Statistics

## Task

Display comprehensive statistics for Anki decks including card counts, study progress, and review forecasts.

## Arguments

- `deck-name` (optional): Specific deck to analyze (default: show all decks summary)
- `--all`: Show detailed stats for all decks
- `--verbose`: Include additional metrics (ease distribution, intervals, etc.)

## Process

### 1. Get Deck List

```
mcp__anki__deckActions with listDecks
```

### 2. Gather Statistics

For each deck (or specified deck):

```
mcp__anki__deckActions with deckStats
```

Query card states:
```
mcp__anki__findNotes with queries:
- "deck:[name] is:new"
- "deck:[name] is:learn"
- "deck:[name] is:review"
- "deck:[name] is:due"
- "deck:[name] is:suspended"
```

### 3. Display Summary

#### All Decks Overview (default)

```
Anki Deck Statistics
════════════════════

Deck                    Total    New    Due    Learning
──────────────────────────────────────────────────────
Default                   150     25     12        5
Programming::Python       280     40     18        8
Programming::JavaScript   195     15     22        3
Languages::Spanish        420     80     45       12
──────────────────────────────────────────────────────
Total                    1045    160     97       28

Due today: 97 cards across 4 decks
```

#### Single Deck Detail

```
Deck: Programming::Python
═════════════════════════

Card Counts
───────────
Total cards:     280
  New:            40 (14%)
  Learning:        8 (3%)
  Young:          82 (29%)
  Mature:        150 (54%)

Suspended:        15
Buried:            2

Today's Progress
────────────────
Due:              18
Reviewed:         12
Remaining:         6
New limit:        20

Study Streak
────────────
Current streak:   14 days
Longest streak:   45 days
Last review:      Today
```

### 4. Verbose Output (if --verbose)

```
Ease Distribution
─────────────────
< 200%:    12 cards (struggling)
200-250%:  45 cards
250-300%: 180 cards (optimal)
> 300%:    43 cards (easy)

Interval Distribution
─────────────────────
< 1 day:    8 cards (learning)
1-7 days:  35 cards
1-4 weeks: 87 cards
1-6 months: 95 cards
> 6 months: 55 cards (mature)

Review Forecast (next 7 days)
─────────────────────────────
Mon:  18 due
Tue:  22 due
Wed:  15 due
Thu:  28 due
Fri:  19 due
Sat:  12 due
Sun:  25 due
```

### 5. Recommendations

Based on statistics, provide actionable insights:

```
Recommendations
───────────────
- 12 cards have ease < 200% - consider reformulating
- 40 new cards waiting - you're keeping up well
- Review load is manageable (~20/day average)
- Consider increasing new cards/day (current: 10)
```

## Output Formats

The command outputs human-readable tables by default.

For programmatic use:
- Statistics are gathered via MCP tools
- Can be formatted as needed

## Error Handling

| Error | Action |
|-------|--------|
| Deck not found | List available decks with fuzzy matches |
| No cards in deck | Report empty deck |
| Connection failed | Check Anki is running |

## Examples

```
/anki/deck-stats
/anki/deck-stats "Programming::Python"
/anki/deck-stats --all --verbose
/anki/deck-stats Languages --verbose
```

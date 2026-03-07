# Anki Query Syntax Reference

Complete reference for Anki search queries used with `findNotes` and `findCards`.

## Basic Searches

### Text Search

| Query | Description |
|-------|-------------|
| `word` | Match word anywhere |
| `"exact phrase"` | Match exact phrase |
| `word*` | Wildcard (prefix match) |
| `_` | Match single character |

### Field-Specific Search

| Query | Description |
|-------|-------------|
| `front:text` | Search Front field |
| `back:text` | Search Back field |
| `fieldname:text` | Search specific field |
| `field:*` | Field is not empty |
| `field:` | Field is empty |

## Deck Filters

| Query | Description |
|-------|-------------|
| `deck:DeckName` | Exact deck match |
| `deck:Parent::Child` | Subdeck |
| `deck:Parent::*` | All subdecks of Parent |
| `"deck:Deck Name"` | Deck with spaces |
| `-deck:Excluded` | Not in deck |

## Tag Filters

| Query | Description |
|-------|-------------|
| `tag:tagname` | Has tag |
| `tag:parent::child` | Hierarchical tag |
| `-tag:tagname` | Missing tag |
| `tag:*` | Has any tag |
| `-tag:*` | No tags |

## Card State Filters

| Query | Description |
|-------|-------------|
| `is:due` | Due for review |
| `is:new` | New/unstudied |
| `is:learn` | In learning phase |
| `is:review` | Review cards |
| `is:suspended` | Suspended |
| `is:buried` | Buried |
| `-is:suspended` | Not suspended |

## Date Filters

| Query | Description |
|-------|-------------|
| `added:N` | Added in last N days |
| `added:1` | Added today |
| `edited:N` | Edited in last N days |
| `rated:N` | Rated in last N days |
| `rated:N:E` | Rated with ease E in N days |
| `introduced:N` | First review in N days |

## Property Filters

| Query | Description |
|-------|-------------|
| `prop:ivl>=N` | Interval >= N days |
| `prop:ivl<N` | Interval < N days |
| `prop:due=N` | Due in N days |
| `prop:due<N` | Due within N days |
| `prop:ease>=N` | Ease factor >= N% |
| `prop:reps>=N` | Review count >= N |
| `prop:lapses>=N` | Lapse count >= N |

## Note Type and Card Filters

| Query | Description |
|-------|-------------|
| `note:Basic` | Note type is Basic |
| `note:Cloze` | Cloze note type |
| `card:1` | First card template |
| `card:Card 2` | Specific template name |
| `nid:123456` | Specific note ID |
| `cid:123456` | Specific card ID |

## Flag Filters

| Query | Description |
|-------|-------------|
| `flag:0` | No flag |
| `flag:1` | Red flag |
| `flag:2` | Orange flag |
| `flag:3` | Green flag |
| `flag:4` | Blue flag |

## Combining Queries

### AND (implicit)

```
deck:Default tag:important is:due
```

All conditions must match.

### OR

```
tag:math OR tag:physics
```

Either condition matches.

### Negation

```
-tag:easy -is:suspended
```

Exclude matching cards.

### Grouping

```
(tag:math OR tag:physics) deck:Science
```

Use parentheses for complex logic.

## Examples

### Find due cards in a deck with tag

```
deck:Languages tag:vocabulary is:due
```

### Find new cards added this week

```
is:new added:7
```

### Find cards with high lapse count

```
prop:lapses>=5 -is:suspended
```

### Find untagged cards

```
-tag:* deck:Default
```

### Find cards needing review soon

```
prop:due<=3 -is:suspended -is:new
```

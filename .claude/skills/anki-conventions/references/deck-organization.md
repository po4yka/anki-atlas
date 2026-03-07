# Deck Organization Reference

Strategies for organizing Anki decks and tags effectively.

## Deck Hierarchy

Anki supports hierarchical decks using `::` separator:

```
Languages
Languages::Spanish
Languages::Spanish::Vocabulary
Languages::Spanish::Grammar
Languages::Japanese
Languages::Japanese::Kanji
Languages::Japanese::Vocabulary
```

### Hierarchy Recommendations

| Level | Max Depth | Example |
|-------|-----------|---------|
| 1 | Subject area | `Languages`, `Programming`, `Medicine` |
| 2 | Topic | `Languages::Spanish`, `Programming::Python` |
| 3 | Subtopic (optional) | `Programming::Python::Async` |

**Keep hierarchy to 2-3 levels maximum.** Deeper nesting creates maintenance burden.

### MCP Limitation

The Anki MCP server (`@ankimcp/anki-mcp-server`) supports max 2 levels: `Parent::Child`. For deeper organization, use tags instead.

## Decks vs Tags Decision Tree

```
Need separate study sessions?
-- Yes -> Use decks
   "I want to study Spanish vocabulary separately"

-- No -> Use tags
    "I want to find all cards about loops"

Need different scheduling settings?
-- Yes -> Use decks
   "Medical cards need 95% retention, casual 85%"

-- No -> Use tags

Will cards belong to multiple categories?
-- Yes -> Use tags (cards can have multiple tags)

-- No -> Either works
```

### Summary Table

| Use Case | Decks | Tags |
|----------|-------|------|
| Different study sessions | Yes | No |
| Different scheduling | Yes | No |
| Multiple categories per card | No | Yes |
| Quick filtering in browser | Either | Either |
| Cross-cutting concerns | No | Yes |
| Source tracking | No | Yes |

## Tag Strategies

### Format

Use `::` as the hierarchy separator with max 2 levels, and kebab-case for words:

```
android::compose          # domain::topic
kotlin::coroutines        # domain::topic
cs::data-structures       # domain::topic
topic::system-design      # cross-cutting theme
difficulty::medium        # difficulty level
source::book-name         # card origin
context::interview-prep   # study context
```

### Status Tracking

Prefer Anki's built-in features over status tags:

| Need | Use | Not |
|------|-----|-----|
| Mark difficult cards | Anki flags (red/orange) | `difficult` tag |
| Track failures | Built-in leech detection | `needs-review` tag |
| Suspend temporarily | Anki suspend feature | `paused` tag |
| New/unverified | `is:new` search filter | `new` tag |

### Source and Context Tags

```
source::effective-kotlin    # Book source
source::android-docs        # Documentation
context::interview-prep     # Study purpose
context::certification      # Study purpose
```

See `tag-conventions.md` for full rules, normalization, and anti-patterns.

## Recommended Limits

| Metric | Recommendation | Why |
|--------|----------------|-----|
| Top-level decks | 5-10 | Easy to navigate |
| Subdeck depth | 2-3 levels | Manageable maintenance |
| Cards per deck | No strict limit | But consider splitting at 5000+ |
| Tags per card | 2-5 | Enough to filter, not overwhelming |

## Daily Limits

Set in deck options:

| Setting | Casual | Active | Intensive |
|---------|--------|--------|-----------|
| New cards/day | 5-10 | 15-25 | 30-50 |
| Reviews/day | 100-200 | 200-500 | No limit |

**Important:** Limits apply to deck and all subdecks combined by default.

## MCP Commands for Deck Management

```
mcp__anki__deckActions
-- listDecks    - View all decks with optional statistics
-- createDeck   - Add new deck (max 2 levels)
-- deckStats    - Get deck statistics
-- changeDeck   - Move cards between decks
```

## Further Reading

See `query-syntax.md` for filtering cards by deck and tag in searches.

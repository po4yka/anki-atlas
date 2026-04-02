# Deck Naming Reference

## Flat Hierarchy Rule

Use ONLY flat deck names. **Never create subdecks**.

| CORRECT | WRONG |
|---------|-------|
| `Kotlin` | `Kotlin::Interview` |
| `Android` | `Android::Compose` |
| `SystemDesign` | `SystemDesign::Concepts` |

## Active Decks

| Deck | ~Notes | Topics |
|------|--------|--------|
| `Algorithms` | 110 | Data structures, algorithms, complexity |
| `Android` | 1476 | Android SDK, Compose, architecture |
| `Backend` | 108 | APIs, databases, system design patterns |
| `CompSci` | 322 | CS fundamentals, OS, networking |
| `KMP` | 210 | Kotlin Multiplatform |
| `Kotlin` | 1002 | Kotlin language, coroutines, flows |
| `SystemDesign` | 213 | Distributed systems, scalability |
| `Когнитивные искажения` | 366 | Cognitive biases (Russian-only deck) |

Total: ~3828 notes across 8 active decks.

**Note**: The cognitive biases deck uses a Cyrillic name. Skills must handle non-ASCII deck names.

## Inactive / Template Decks

Exclude from card generation:

| Deck | Purpose |
|------|---------|
| `Default` | Anki default (empty) |
| `APF (3.0.0) — Demo` | APF format demo cards |
| `Memrise cards [template]` | Import template |

## CLI Usage

```bash
# Correct - flat deck name
cargo run --bin anki-atlas -- search "query" --deck "Kotlin"

# Correct - Cyrillic deck name (quote it)
cargo run --bin anki-atlas -- search "query" --deck "Когнитивные искажения"

# Wrong - subdeck hierarchy
cargo run --bin anki-atlas -- search "query" --deck "Kotlin::Interview"
```

Deck is created automatically if it doesn't exist.

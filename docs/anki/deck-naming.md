# Deck Naming Reference

## Flat Hierarchy Rule

Use ONLY flat deck names. **Never create subdecks**.

| CORRECT | WRONG |
|---------|-------|
| `Kotlin` | `Kotlin::Interview` |
| `Android` | `Android::Compose` |
| `SystemDesign` | `SystemDesign::Concepts` |

## Allowed Decks

| Deck | Topics |
|------|--------|
| `Algorithms` | Data structures, algorithms, complexity |
| `Android` | Android SDK, Compose, architecture |
| `Backend` | APIs, databases, system design patterns |
| `CompSci` | CS fundamentals, OS, networking |
| `Kotlin` | Kotlin language, coroutines, flows |
| `SystemDesign` | Distributed systems, scalability |

## CLI Usage

```bash
# Correct - flat deck name
cargo run --bin anki-atlas -- search "query" --deck "Kotlin"

# Wrong - subdeck hierarchy
cargo run --bin anki-atlas -- search "query" --deck "Kotlin::Interview"
```

Deck is created automatically if it doesn't exist.

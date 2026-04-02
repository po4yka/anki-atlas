# Tag Taxonomy Reference

## Required Tags

Every card MUST have:
1. **One difficulty tag**: `difficulty::easy`, `difficulty::medium`, or `difficulty::hard`
2. **One or more topic tags**: from the taxonomy below

## Format Rules

| Rule | Convention | Example |
|------|-----------|---------|
| Hierarchy separator | `::` (double-colon) | `kotlin::coroutines` |
| Word separator | `-` (kebab-case) | `state-management` |
| Code identifiers | Original casing | `ArrayList`, `WorkManager` |
| Max depth | 2 levels (`prefix::topic`) | `kotlin::coroutines` |
| Case | lowercase (except code IDs) | `cs::algorithms` |

## Domain Prefixes (16)

All 16 valid prefixes (source: `crates/taxonomy/src/tags.rs` `VALID_PREFIXES`):

| Prefix | Scope | Examples |
|--------|-------|---------|
| `android::` | Android framework, Jetpack, tooling | `android::compose`, `android::lifecycle`, `android::room` |
| `kotlin::` | Kotlin language features | `kotlin::coroutines`, `kotlin::flow`, `kotlin::null-safety` |
| `cs::` | Computer science fundamentals | `cs::algorithms`, `cs::data-structures`, `cs::concurrency` |
| `topic::` | High-level cross-cutting themes | `topic::system-design`, `topic::security`, `topic::patterns` |
| `difficulty::` | Card difficulty level | `difficulty::easy`, `difficulty::medium`, `difficulty::hard` |
| `lang::` | Natural language | `lang::en`, `lang::ru` |
| `source::` | Card origin tracking | `source::book-name`, `source::course-name` |
| `context::` | Study context | `context::interview-prep`, `context::certification` |
| `bias::` | Cognitive biases | (used with Cognitive Biases deck) |
| `testing::` | Testing practices | `testing::best-practices` |
| `architecture::` | Architecture patterns | `architecture::philosophy` |
| `performance::` | Performance topics | `performance::rendering` |
| `platform::` | Platform-specific | `platform::android` |
| `security::` | Security topics | `security::privacy` |
| `networking::` | Networking topics | `networking::http` |
| `skill::` | Skill-level meta | (defined in code, not yet in live data) |

## Kotlin Tags

40 `kotlin::` tags in live collection. Representative tags:

| Tag | Use For |
|-----|---------|
| `kotlin::coroutines` | suspend, CoroutineScope, launch, async, Job |
| `kotlin::flow` | Flow, StateFlow, SharedFlow, operators |
| `kotlin::channels` | Channel, produce, consume, pipelines |
| `kotlin::dispatchers` | Dispatchers.IO, Default, Main |
| `kotlin::collections` | List, Map, Set, Sequence, fold, filter |
| `kotlin::classes` | data class, sealed class, enum, object |
| `kotlin::types` | generics, variance, null-safety |
| `kotlin::functions` | lambdas, inline, extension, scope functions |
| `kotlin::general` | basic language features, misc |

## Android Tags

132 `android::` tags in live collection. Representative tags:

| Tag | Use For |
|-----|---------|
| `android::lifecycle` | Activity/Fragment lifecycle |
| `android::activity` | Launch modes, task management |
| `android::compose` | Composables, recomposition, state |
| `android::viewmodel` | ViewModel scope, SavedStateHandle |
| `android::architecture` | MVVM, MVP, MVI, Clean Architecture |
| `android::room` | Database, Entity, DAO |
| `android::testing` | UI tests, instrumentation |
| `android::general` | Android basics, misc |

## CS Tags

19 `cs::` tags in live collection:

| Tag | Use For |
|-----|---------|
| `cs::algorithms` | Sorting, searching, graph algorithms |
| `cs::data-structures` | Trees, heaps, hash maps |
| `cs::concurrency` | Threads, locks, synchronization |
| `cs::patterns` | Design patterns (GoF, etc.) |
| `cs::architecture` | Software architecture |
| `cs::oop` | Object-oriented principles |
| `cs::database` | SQL, indexing, transactions |
| `cs::networking` | Protocols, sockets, HTTP |
| `cs::os` | Operating systems, processes |
| `cs::fp` | Functional programming |
| `cs::type-systems` | Type theory, type safety |
| `cs::testing` | Testing methodology |
| `cs::security` | Cryptography, auth |
| `cs::system-design` | Distributed systems |
| `cs::compilers` | Parsing, compilation |
| `cs::general` | CS fundamentals, misc |

## Bare Tags (Legacy)

1486 of 1703 total tags lack a `::` prefix (e.g., `algorithms`, `coroutines`, `batch-1`, `code-quality`). These are **legacy tags** from before the prefix convention was adopted.

- The normalization map (`TAG_MAP` in `crates/taxonomy/src/map.rs`, 464 entries) maps common bare tags to canonical prefixed forms
- **New cards MUST use prefixed tags** — never add bare tags
- Run `cargo run --bin anki-atlas -- tag-audit tags.txt --fix` to normalize existing bare tags
- Some bare tags coexist with their prefixed equivalent (e.g., `algorithms` + `cs::algorithms`)

## Special Tags

| Tag | Use For |
|-----|---------|
| `atomic` | Cards split from long cards |

## Validation

Use `crates/taxonomy` for programmatic tag validation:
- `lookup_tag()` — resolve tag via TAG_MAP
- `is_known_topic_tag()` — check if tag is canonical
- `VALID_PREFIXES` — all 16 accepted prefixes

CLI: `cargo run --bin anki-atlas -- tag-audit tags.txt --fix`

## Examples

```yaml
# Kotlin card
Tags: kotlin::coroutines, kotlin::flow, difficulty::medium, lang::en

# Android card
Tags: android::lifecycle, android::activity, difficulty::medium, lang::en

# Android + Kotlin card
Tags: android::viewmodel, kotlin::coroutines, difficulty::hard, lang::en

# CS card
Tags: cs::patterns, cs::oop, difficulty::medium, lang::en

# Cognitive bias card (Russian)
Tags: bias::anchoring, difficulty::easy, lang::ru
```

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

## Domain Prefixes

| Prefix | Scope | Examples |
|--------|-------|---------|
| `android::` | Android framework, Jetpack, tooling | `android::compose`, `android::lifecycle`, `android::room` |
| `kotlin::` | Kotlin language features | `kotlin::coroutines`, `kotlin::flow`, `kotlin::null-safety` |
| `cs::` | Computer science fundamentals | `cs::algorithms`, `cs::data-structures`, `cs::concurrency` |
| `topic::` | High-level cross-cutting themes | `topic::system-design`, `topic::security`, `topic::patterns` |
| `difficulty::` | Card difficulty level | `difficulty::easy`, `difficulty::medium`, `difficulty::hard` |
| `lang::` | Natural/programming language | `lang::en`, `lang::ru`, `lang::kotlin` |
| `source::` | Card origin tracking | `source::book-name`, `source::course-name` |
| `context::` | Study context | `context::interview-prep`, `context::certification` |

## Kotlin Tags

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

| Tag | Use For |
|-----|---------|
| `android::lifecycle` | Activity/Fragment lifecycle |
| `android::activities` | Launch modes, task management |
| `android::compose` | Composables, recomposition, state |
| `android::viewmodel` | ViewModel scope, SavedStateHandle |
| `android::architecture` | MVVM, MVP, MVI, Clean Architecture |
| `android::room` | Database, Entity, DAO |
| `android::general` | Android basics, misc |

## Special Tags

| Tag | Use For |
|-----|---------|
| `atomic` | Cards split from long cards |

## Validation

Use `packages.taxonomy` for programmatic tag validation:
- `validate_tag()` - check tag format
- `normalize_tag()` - fix common issues
- `suggest_tag()` - suggest corrections

CLI: `uv run anki-atlas tag-audit tags.txt --fix`

## Examples

```yaml
# Kotlin card
Tags: kotlin::coroutines, kotlin::flow, difficulty::medium

# Android card
Tags: android::lifecycle, android::activities, difficulty::medium

# Android + Kotlin card
Tags: android::viewmodel, kotlin::coroutines, difficulty::hard
```

# Tag Conventions

Single source of truth for Anki tag naming, hierarchy, and validation.

## Format Rules

| Rule | Convention | Example |
|------|-----------|---------|
| Hierarchy separator | `::` (double-colon) | `android::compose` |
| Word separator | `-` (kebab-case) | `state-management` |
| Code identifiers | Original casing | `ArrayList`, `WorkManager` |
| Max depth | 2 levels (`prefix::topic`) | `kotlin::coroutines` |
| Case | lowercase (except code IDs) | `cs::algorithms` |

## Domain Prefixes

### Primary Domains

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

### Minor Domains

| Prefix | Scope | Examples |
|--------|-------|---------|
| `testing::` | Testing practices | `testing::best-practices` |
| `architecture::` | Architecture philosophy | `architecture::philosophy` |
| `performance::` | Performance topics | `performance::rendering` |
| `platform::` | Platform-specific | `platform::android` |
| `security::` | Security topics | `security::privacy` |
| `networking::` | Networking topics | `networking::http` |

## Irreducible Tags

These tags follow special formats that must be preserved as-is:

| Pattern | Count | Purpose |
|---------|-------|---------|
| `slug:bias-*` | 366 | Cognitive bias card identifiers (Russian deck) |
| `bias_*` | 12 | Cognitive bias category groupings |

Examples: `slug:bias-anchoring-def-ru`, `bias_decisions`, `bias_social`

## Code Identifiers

Tags representing API names, class names, or framework types preserve original casing:

`ArrayList`, `LinkedList`, `Material3`, `WorkManager`, `SQLDelight`

These are never lowercased or kebab-cased.

## Unprefixed Tags

General-purpose concept tags using kebab-case:

`dependency-injection`, `state-management`, `binary-search`, `clean-architecture`

When creating new cards, prefer domain-prefixed tags for categorization. Unprefixed tags are acceptable for concepts that span multiple domains.

## Normalization Rules

| Input | Output | Rule |
|-------|--------|------|
| `cs_algorithms` | `cs::algorithms` | Replace `_` prefix separator with `::` |
| `my_tag` | `my-tag` | Replace `_` word separator with `-` |
| `android/compose` | `android::compose` | Replace `/` with `::` |
| `Programming::Python` | `programming::python` | Lowercase (unless code ID) |
| `android::compose::animation::advanced` | `android::compose-animation` | Flatten to max 2 levels |
| `ArrayList` | `ArrayList` | Preserve code identifier casing |

Programmatic validation: `packages.taxonomy.validate_tag()`, `packages.taxonomy.normalize_tag()`

CLI: `uv run anki-atlas tag-audit tags.txt --fix`

## Anti-Patterns

| Anti-Pattern | Problem | Correct Form |
|-------------|---------|--------------|
| `cs_algorithms` | Underscore as prefix separator | `cs::algorithms` |
| `my_custom_tag` | Underscores in words | `my-custom-tag` |
| `android/compose` | Slash as hierarchy separator | `android::compose` |
| `Android::Compose` | Uppercase prefix | `android::compose` |
| `a::b::c::d` | Too deep (>2 levels) | Flatten: `a::b-c-d` or `a::b` |
| `new`, `verified` | Status tags (use Anki flags) | Remove; use built-in flags |
| `batch-1`, `phase4` | Process tags | Remove entirely |

## Validation Checklist

When creating or reviewing tags:

1. [ ] Uses `::` for domain prefix (not `_`, `/`, or `.`)
2. [ ] Uses `-` between words (not `_` or camelCase)
3. [ ] Max 2 hierarchy levels (`prefix::topic`)
4. [ ] Lowercase except code identifiers
5. [ ] Has domain prefix for categorizable concepts
6. [ ] No status/process tags (use Anki flags instead)
7. [ ] Code identifiers preserved in original casing
8. [ ] Irreducible tags (`slug:*`, `bias_*`) left unchanged

## Tag Selection Strategy

When tagging a new card:

1. **Domain prefix first**: Pick the primary domain (`android::`, `kotlin::`, `cs::`)
2. **1-2 concept tags**: Specific topic within the domain (`coroutines`, `state-management`)
3. **Difficulty** (optional): `difficulty::easy`, `difficulty::medium`, `difficulty::hard`
4. **Source** (optional): `source::book-name` if from a specific source
5. **Context** (optional): `context::interview-prep` if study-context-specific

Aim for 2-5 tags per card. More than 5 usually indicates the card covers too many concepts.

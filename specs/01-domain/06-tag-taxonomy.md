# Spec 06: Tag Taxonomy

## Goal

Migrate the tag taxonomy system (tag mapping, normalization, validation) into `packages/taxonomy/`.

## Source

- `/Users/npochaev/GitHub/ai-agent-anki/TAGS.md` -- Complete tag inventory (1,570 tags organized by prefix: `android::`, `kotlin::`, `cs::`, `topic::`, `difficulty::`, `lang::`, etc.)
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/utils/tag_taxonomy.py` -- `TAG_MAPPING` dict, `normalize_tag()` function, domain-specific normalization rules (32KB)
- `/Users/npochaev/GitHub/ai-agent-anki/.claude/skills/anki-conventions/references/tag-conventions.md` -- Tag naming rules and conventions

## Target

### `packages/taxonomy/tags.py` (NEW)

- `TagPrefix` enum -- standard prefixes (`android`, `kotlin`, `cs`, `topic`, `difficulty`, `lang`)
- `TAG_MAPPING: dict[str, str]` -- canonical mapping from variants to normalized tags
- `VALID_PREFIXES: frozenset[str]` -- recognized double-colon prefixes

### `packages/taxonomy/normalize.py` (NEW)

- `normalize_tag(tag: str) -> str` -- normalize a single tag to canonical form
- `validate_tag(tag: str) -> list[str]` -- return list of validation issues (empty if valid)
- `suggest_tag(input_tag: str) -> list[str]` -- fuzzy match against known tags
- Tag rules to enforce:
  - Double-colon `::` for domain prefixes
  - Kebab-case within tags
  - No trailing/leading whitespace
  - No duplicate separators

### `packages/taxonomy/__init__.py` (UPDATE)

Re-export: `normalize_tag`, `validate_tag`, `TAG_MAPPING`, `TagPrefix`

## Acceptance Criteria

- [ ] `packages/taxonomy/tags.py` contains `TAG_MAPPING`, `TagPrefix`, `VALID_PREFIXES`
- [ ] `packages/taxonomy/normalize.py` contains `normalize_tag`, `validate_tag`, `suggest_tag`
- [ ] `from packages.taxonomy import normalize_tag, TAG_MAPPING` works
- [ ] Normalization handles: case folding, prefix standardization, separator normalization
- [ ] Tests in `tests/test_taxonomy.py` cover: normalization edge cases, validation rules, prefix handling
- [ ] `make check` passes

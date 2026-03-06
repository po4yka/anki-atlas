# Spec 08: Card Registry

## Goal

Migrate the SQLite-based card registry (local tracking of cards, notes, mappings) into `packages/card/registry.py`.

## Source

- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/utils/card_registry.py` -- `CardRegistry` (SQLite), `CardEntry`, `NoteEntry`, schema management (53KB)
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/services/mapping_service.py` -- `MappingService`, `CardMappingEntry`, `NoteMapping` (12KB)

## Target

### `packages/card/registry.py` (NEW)

- `CardRegistry` -- SQLite-based card tracking
  - Card CRUD: `add_card()`, `get_card()`, `update_card()`, `delete_card()`, `find_cards()`
  - Note tracking: `add_note()`, `get_note()`, `list_notes()`
  - Mapping: `get_mapping()`, `update_mapping()`
  - Stats: `card_count()`, `note_count()`
- `CardEntry` -- frozen dataclass for card records
- `NoteEntry` -- frozen dataclass for note records
- Schema versioning with migrations

### `packages/card/mapping.py` (NEW)

- `CardMappingEntry` -- mapping metadata for single card
- `NoteMapping` -- mapping metadata for single note
- Mapping functions for card <-> note <-> Anki note relationships

The source is 53KB -- split into registry.py (core CRUD) and mapping.py (mapping logic). Keep under 600 lines each.

## Acceptance Criteria

- [ ] `packages/card/registry.py` contains `CardRegistry`, `CardEntry`, `NoteEntry`
- [ ] `packages/card/mapping.py` contains `CardMappingEntry`, `NoteMapping`
- [ ] SQLite operations use parameterized queries (no SQL injection)
- [ ] `from packages.card import CardRegistry` works
- [ ] Tests in `tests/test_card_registry.py` cover: CRUD operations, schema migration, mapping lookups (in-memory SQLite)
- [ ] `make check` passes

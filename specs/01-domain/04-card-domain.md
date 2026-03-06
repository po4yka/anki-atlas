# Spec 04: Card Domain Models

## Goal

Migrate card domain entities and slug service into `packages/card/`.

## Source

- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/domain/entities/card.py` -- `Card`, `CardManifest`, `SyncAction`, `SyncActionType`, `CardValidationError`, `ManifestValidationError`, `VALID_NOTE_TYPES`, `VALID_LANGUAGES`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/domain/services/slug_service.py` -- `SlugService` (static methods: `slugify`, `generate_slug`, `compute_hash`, `generate_deterministic_guid`, `extract_components`, `is_valid_slug`, `compute_content_hash`, `compute_metadata_hash`)
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/domain/entities/__init__.py` -- full entity exports list

## Target

### `packages/card/models.py` (NEW)

Migrate core card domain entities:
- `Card` -- frozen dataclass representing an Anki flashcard
- `CardManifest` -- value object linking card to source note
- `SyncAction` -- domain entity for sync operations
- `SyncActionType` -- enum (CREATE, UPDATE, DELETE, SKIP)

Adapt:
- Replace `from src.X` imports with `from packages.X`
- Use `Language` enum from `packages.common.types` instead of `VALID_LANGUAGES` frozenset
- Use exceptions from `packages.common.exceptions`
- Add `from __future__ import annotations`

### `packages/card/slug.py` (NEW)

Migrate `SlugService` with all its static methods. This is a pure utility class with no I/O.

### `packages/card/__init__.py` (UPDATE)

Re-export key types: `Card`, `CardManifest`, `SyncAction`, `SyncActionType`, `SlugService`

## Acceptance Criteria

- [ ] `packages/card/models.py` contains `Card`, `CardManifest`, `SyncAction`, `SyncActionType`
- [ ] `packages/card/slug.py` contains `SlugService` with all static methods
- [ ] All models use frozen dataclasses
- [ ] Imports use `packages.common.types` and `packages.common.exceptions`
- [ ] `from packages.card import Card, SlugService` works
- [ ] Tests in `tests/test_card_models.py` cover: Card creation, slug generation, content hash, manifest validation
- [ ] `make check` passes

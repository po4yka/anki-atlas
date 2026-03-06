# Spec 07: AnkiConnect Client

## Goal

Migrate the AnkiConnect HTTP client into `packages/anki/connect.py`, unifying two implementations.

## Source

Primary (async, more complete):
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/utils/anki_connect.py` -- `AnkiConnectClient` (async httpx), `AnkiConnectError`, `NoteInfo` dataclass (15KB)

Secondary (sync + async):
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/anki/client.py` -- AnkiConnect client
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/anki/services/` -- `anki_card_service.py`, `anki_deck_service.py`, `anki_note_service.py`, `anki_tag_service.py`, `anki_media_service.py`, `anki_model_service.py`, `anki_http_client.py`

## Target

### `packages/anki/connect.py` (NEW)

Unified AnkiConnect client:
- `AnkiConnectClient` -- async HTTP client using httpx
  - Core methods: `invoke(action, **params)`, `ping()`, `version()`
  - Note methods: `add_note()`, `update_note_fields()`, `delete_notes()`, `find_notes()`, `notes_info()`
  - Deck methods: `deck_names()`, `create_deck()`, `delete_decks()`
  - Tag methods: `get_tags()`, `add_tags()`, `remove_tags()`
  - Model methods: `model_names()`, `model_field_names()`
  - Sync method: `sync()`
- Use `AnkiConnectError` from `packages.common.exceptions`
- Configurable base URL (default: `http://localhost:8765`)

Study both implementations. The claude-code version is simpler; the obsidian-to-anki version has richer service decomposition. Consolidate into a single client with clear method grouping.

## Acceptance Criteria

- [ ] `packages/anki/connect.py` contains `AnkiConnectClient` with async methods
- [ ] Uses httpx for HTTP, structlog for logging
- [ ] Uses `AnkiConnectError` from `packages.common.exceptions`
- [ ] `from packages.anki.connect import AnkiConnectClient` works
- [ ] Tests in `tests/test_anki_connect.py` cover: invoke, note CRUD, deck operations (mock httpx)
- [ ] `make check` passes

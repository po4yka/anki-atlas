# Spec 15: Sync Engine

## Goal

Extend the existing `packages/anki/sync.py` with the more comprehensive sync engine from obsidian-to-anki.

## Source

- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/sync/engine.py` -- `SyncEngine` (main orchestrator)
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/sync/state_db.py` -- `StateDB` (SQLite WAL, implements IStateRepository)
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/sync/recovery.py` -- `CardRecovery`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/sync/transactions.py` -- `CardTransaction`, `RollbackAction`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/sync/progress.py` -- `ProgressTracker`, `SyncProgress`, `SyncPhase`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/sync/change_applier/applier.py` -- `ChangeApplier`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/sync/anki_state_manager.py` -- `AnkiStateManager`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/sync/indexer.py` -- `VaultIndexer`, `AnkiIndexer`, `SyncIndexer`

Existing in anki-atlas:
- `packages/anki/sync.py` -- existing two-way sync logic (11KB)

## Target

### `packages/anki/sync/` directory (convert existing sync.py to package):

Rename existing `packages/anki/sync.py` to `packages/anki/sync/core.py` and create:

- `core.py` -- existing anki-atlas sync logic (renamed from sync.py)
- `engine.py` -- `SyncEngine` orchestrator from obsidian-to-anki
- `state.py` -- `StateDB` for persistent sync state (SQLite WAL)
- `progress.py` -- `ProgressTracker`, `SyncProgress`, `SyncPhase`
- `recovery.py` -- `CardRecovery`, `CardTransaction` for rollback
- `__init__.py` -- re-export from both core.py (existing) and new modules

**Important:** The existing `packages/anki/sync.py` has imports throughout the codebase. After converting to `packages/anki/sync/`, ensure `packages/anki/sync/__init__.py` re-exports everything that was in the original `sync.py` so existing imports don't break.

## Acceptance Criteria

- [ ] `packages/anki/sync/` is a package with core.py, engine.py, state.py, progress.py, recovery.py
- [ ] All existing imports of `packages.anki.sync` still work (re-exports in `__init__.py`)
- [ ] `SyncEngine` orchestrates the full sync lifecycle
- [ ] `StateDB` uses SQLite WAL with parameterized queries
- [ ] Existing tests in `tests/` still pass unchanged
- [ ] Tests in `tests/test_sync_engine.py` cover: engine lifecycle, state persistence, recovery
- [ ] `make check` passes

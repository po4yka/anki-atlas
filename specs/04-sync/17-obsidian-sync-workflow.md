# Spec 17: Obsidian Sync Workflow

## Goal

Wire together obsidian parsing, card generation, and Anki sync into a complete workflow.

## Source

- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/sync/note_scanner.py` -- `NoteScanner`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/sync/note_processor.py` -- `SingleNoteProcessor`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/sync/parallel_processor.py` -- `ParallelNoteProcessor`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/sync/card_generator.py` -- `CardGenerator` (sync-layer wrapper)
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/application/services/sync_orchestrator.py` -- `SyncOrchestrator`, `SyncResult`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/application/use_cases/sync_notes.py`

## Target

### `packages/obsidian/sync.py` (NEW)

High-level workflow that connects:
1. `packages/obsidian/parser.py` -- discover and parse notes
2. `packages/generator/` -- generate cards from parsed content
3. `packages/validation/` -- validate generated cards
4. `packages/anki/sync/` -- sync cards to Anki

Classes:
- `ObsidianSyncWorkflow` -- main orchestrator
  - `scan_vault(vault_path) -> list[ParsedNote]`
  - `process_note(note: ParsedNote) -> list[GeneratedCard]`
  - `sync_cards(cards: list[GeneratedCard]) -> SyncResult`
  - `run(vault_path) -> SyncResult` -- full pipeline
- `SyncResult` -- frozen dataclass with created/updated/skipped/failed counts

### Design:

- The workflow is a thin orchestrator -- it delegates to package modules
- Each step is independently testable
- Supports incremental sync (only changed notes)
- Progress reporting via callbacks

## Acceptance Criteria

- [ ] `packages/obsidian/sync.py` contains `ObsidianSyncWorkflow`, `SyncResult`
- [ ] Workflow connects parser -> generator -> validator -> sync
- [ ] Each step can be used independently
- [ ] `from packages.obsidian.sync import ObsidianSyncWorkflow` works
- [ ] Tests in `tests/test_obsidian_sync.py` cover: workflow orchestration (mock dependencies)
- [ ] `make check` passes

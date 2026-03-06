# Spec 18: CLI Commands

## Goal

Add new CLI commands to the existing `apps/cli/` for card generation, validation, obsidian sync, and tag audit.

## Source

obsidian-to-anki CLI commands:
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/cli.py` -- main CLI app
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/cli_commands/generate_handler.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/cli_commands/sync_handler.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/cli_commands/validate_commands.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/cli_commands/rag_commands.py`

claude-code-obsidian-anki CLI:
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/cli.py` -- CLI entry point (78KB)

Existing in anki-atlas:
- `apps/cli/__init__.py` -- existing Typer CLI (22KB)

## Target

### `apps/cli/` (EXTEND existing `__init__.py` or add submodules)

Add new commands to the existing CLI app:

1. **`anki-atlas generate`** -- Generate cards from text/notes
   - `--input` -- input file or text
   - `--model` -- LLM model to use
   - `--deck` -- target deck
   - `--dry-run` -- preview without syncing

2. **`anki-atlas validate`** -- Validate cards
   - `--deck` -- deck to validate
   - `--quality` -- run quality scoring
   - `--fix` -- attempt auto-fixes

3. **`anki-atlas obsidian-sync`** -- Sync Obsidian vault to Anki
   - `--vault` -- vault path
   - `--incremental` -- only changed notes
   - `--dry-run` -- preview changes

4. **`anki-atlas tag-audit`** -- Audit tag taxonomy
   - `--fix` -- normalize violations
   - `--report` -- generate report

If `apps/cli/__init__.py` is already large (22KB), create submodules:
- `apps/cli/generate.py`
- `apps/cli/validate.py`
- `apps/cli/obsidian.py`
- `apps/cli/tags.py`

Register subcommand groups in the main `__init__.py`.

## Acceptance Criteria

- [ ] Four new commands available: `generate`, `validate`, `obsidian-sync`, `tag-audit`
- [ ] Commands use packages from `packages/` (generator, validation, obsidian, taxonomy)
- [ ] `uv run anki-atlas --help` shows new commands
- [ ] Existing CLI commands still work
- [ ] Tests in `tests/test_cli_new.py` cover: command registration, help output, basic invocation
- [ ] `make check` passes

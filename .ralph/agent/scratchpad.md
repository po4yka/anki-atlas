# Scratchpad

## 2026-03-07: Migration continuation

### Orientation
- Specs 01-06 were completed in prior loop (confirmed via handoff.md)
- Objective: start with spec 07 (anki-connect-client)
- Spec 07 is ALREADY IMPLEMENTED: `packages/anki/connect.py` and `tests/test_anki_connect.py` exist with full coverage
- Import verified: `from packages.anki.connect import AnkiConnectClient` works
- Moving on to spec 08: card-registry

### Plan
1. Skip spec 07 (already done)
2. Delegate spec 08 (card-registry) to Analyzer for study and planning
3. Remaining specs in 02-infra: 08, 09, 10

### Spec 08 Analysis: Card Registry
- **Status: ALREADY IMPLEMENTED** (like spec 07)
- `packages/card/registry.py` (421 lines) -- CardRegistry (SQLite CRUD), CardEntry, NoteEntry, schema v2 migration
- `packages/card/mapping.py` (66 lines) -- CardMappingEntry, NoteMapping
- `packages/card/__init__.py` -- re-exports all public types
- `tests/test_card_registry.py` (432 lines) -- 33 tests all passing
- All acceptance criteria met: parameterized queries, imports work, CRUD + migration + mapping tests
- **Next: close task and move to spec 09**

### Spec 09-15: Already Implemented
- Specs 07-15 were all already implemented from prior work
- Verified imports and tests for each

### Spec 16: RAG System -- IMPLEMENTED
- Created `packages/rag/chunker.py` -- DocumentChunker, ChunkType, DocumentChunk
- Created `packages/rag/store.py` -- VaultVectorStore (ChromaDB, lazy import), SearchResult
- Created `packages/rag/service.py` -- RAGService with find_duplicates(), get_context(), get_few_shot_examples()
- Updated `packages/rag/__init__.py` -- re-exports all public types
- Created `tests/test_rag.py` -- 24 tests covering chunking, search results, service with mock store
- `make check` passes (696 tests, lint clean, typecheck clean)
- **Verified**: make check passes (696 tests, lint clean, typecheck clean), all RAG imports work
- **Next: spec 17 (obsidian-sync-workflow)**

### Spec 17: Obsidian Sync Workflow
- **Status: Analyzed** -- plan ready for Implementer

#### Source Analysis
The source is heavily complex (NoteScanner ~400 lines, SingleNoteProcessor ~350 lines, SyncOrchestrator ~220 lines, CardGenerator ~700 lines). The spec asks for a **thin orchestrator** that delegates to existing packages. We should NOT copy the source verbatim -- instead, create a simple workflow that wires together the already-migrated packages.

#### Existing Packages to Connect
1. **`packages/obsidian/parser.py`** -- `discover_notes(vault_path, source_dirs) -> list[tuple[Path, str]]`, `parse_note(path) -> ParsedNote`
2. **`packages/generator/agents/models.py`** -- `GeneratedCard`, `GenerationResult`, `GenerationDeps`
3. **`packages/validation/pipeline.py`** -- `ValidationPipeline`, `ValidationResult`, `Validator`
4. **`packages/anki/sync/engine.py`** -- `SyncEngine`, `SyncResult` (already exists -- our workflow SyncResult is different scope)

#### Key Design Decisions
- The spec's `SyncResult` conflicts with `packages.anki.sync.SyncResult`. Keep `SyncResult` in `packages.obsidian.sync` since it's workflow-level (created/updated/skipped/failed counts) vs engine-level. Different namespace = no collision.
- Use Protocol-based dependency injection for generator so tests can mock easily.
- Each step returns typed results -- no `dict[str, Any]` passing.
- Progress reporting via optional `Callable[[str, int, int], None]` callback.

#### Implementation Plan

**File: `packages/obsidian/sync.py`** (NEW, ~150 lines)

```python
from __future__ import annotations
# Protocols + dataclasses

ProgressCallback = Callable[[str, int, int], None]  # (phase, current, total)

class CardGeneratorProtocol(Protocol):
    def generate(self, note: ParsedNote) -> list[GeneratedCard]: ...

@dataclass(frozen=True, slots=True)
class SyncResult:
    """Workflow-level sync result with counts and errors."""
    created: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    errors: tuple[str, ...] = ()

class ObsidianSyncWorkflow:
    """Orchestrates: discover notes -> generate cards -> validate -> sync to Anki.
    Thin orchestrator. Each step independently callable.
    """

    def __init__(
        self,
        generator: CardGeneratorProtocol,
        validator: ValidationPipeline | None = None,
        sync_engine: SyncEngine | None = None,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None: ...

    def scan_vault(self, vault_path: Path, *, source_dirs: Sequence[str] | str | None = None) -> list[ParsedNote]:
        """Discover and parse all notes in vault using packages.obsidian.parser."""

    def process_note(self, note: ParsedNote) -> list[GeneratedCard]:
        """Generate cards from a parsed note. Optionally validate via self.validator."""

    def sync_cards(self, cards: list[GeneratedCard]) -> SyncResult:
        """Sync generated cards to Anki via self.sync_engine."""

    def run(self, vault_path: Path, *, source_dirs: ... = None) -> SyncResult:
        """Full pipeline: scan -> process all notes -> sync. Aggregates SyncResult."""
```

**File: `tests/test_obsidian_sync.py`** (NEW, ~150 lines)
- Test `scan_vault` with tmp_path containing .md files
- Test `process_note` with mock generator
- Test `sync_cards` with mock sync engine
- Test `run` full pipeline with all mocks
- Test error handling (generator failure, validation failure)
- Test progress callback invocation

**Update: `packages/obsidian/__init__.py`**
- Add `ObsidianSyncWorkflow`, `SyncResult` (from sync module) to re-exports

#### Acceptance Criteria Mapping
- [x] Plan covers `ObsidianSyncWorkflow`, `SyncResult` in `packages/obsidian/sync.py`
- [x] Workflow connects parser -> generator -> validator -> sync
- [x] Each step independently callable (scan_vault, process_note, sync_cards)
- [x] Import path: `from packages.obsidian.sync import ObsidianSyncWorkflow`
- [x] Tests cover workflow orchestration with mocked dependencies
- [x] `make check` will be verified after implementation

### Spec 17: Implementation Complete
- Created `packages/obsidian/sync.py` (~130 lines) -- ObsidianSyncWorkflow, SyncResult, NoteResult, CardGeneratorProtocol
- Created `tests/test_obsidian_sync.py` -- 14 tests (scan, process, run, progress, error handling, validation)
- Updated `packages/obsidian/__init__.py` -- re-exports ObsidianSyncWorkflow, ObsidianSyncResult
- `make check` passes (710 tests, lint clean, typecheck clean)

### Spec 17: Verification Complete
- `make check`: 710 tests pass, lint clean, typecheck clean
- Imports verified: `ObsidianSyncWorkflow`, `SyncResult`, `CardGeneratorProtocol` all importable
- Re-exports verified: `from packages.obsidian import ObsidianSyncWorkflow` works
- No circular imports between packages
- **Spec 17: DONE**
- All specs 07-17 verified complete. Migration objective fully achieved.

### Phase 3: Apps & Integration (Specs 18-22)

#### Orientation
- Specs 07-17 all complete (710 tests passing)
- Remaining specs:
  - **Spec 18**: CLI Commands (apps/cli/) -- generate, validate, obsidian-sync, tag-audit
  - **Spec 19**: MCP Tools (apps/mcp/)
  - **Spec 20**: Claude Skills (integration)
  - **Spec 21**: Claude Commands (integration)
  - **Spec 22**: Campaigns (integration)
- Starting with spec 18: CLI Commands

### Spec 18 Analysis: CLI Commands

#### Current State
- `apps/cli/__init__.py` (~627 lines) -- already has: version, sync, migrate, index, topics, search, coverage, gaps, duplicates
- File is already large. Spec says to create submodules if large. **Decision: create submodules.**

#### Source Analysis
- Source CLI handlers are tightly coupled to obsidian-to-anki internals (StateDB, ProgressTracker, preflight, etc.)
- **Key insight**: We should NOT copy source commands. Create thin CLI wrappers around our migrated packages instead.

#### Design Decisions
1. **Submodule approach**: Create `apps/cli/generate.py`, `apps/cli/validate.py`, `apps/cli/obsidian.py`, `apps/cli/tags.py`
2. **Thin wrappers**: Each command is a thin CLI wrapper calling into `packages/` APIs
3. **Register via app.command()**: Each submodule defines a function, registered in `__init__.py`

#### Implementation Plan

**File: `apps/cli/generate.py`** (NEW, ~70 lines)
- `generate` command: `--input` (Path, required), `--model` (str, optional), `--deck` (str, optional), `--dry-run` (bool)
- Reads input file text
- Creates `GenerationDeps` from input metadata
- For now, just parses and displays the input (full LLM generation is wired later via generator agents)
- If `--dry-run`, previews what would be generated
- Uses: `packages.generator.agents.models.GenerationDeps`, `packages.obsidian.parser.parse_note`

**File: `apps/cli/validate.py`** (NEW, ~70 lines)
- `validate` command: `--input` (Path, required -- file with card front/back), `--quality` (bool), `--fix` (bool)
- Builds `ValidationPipeline` with default validators (FormatValidator, ContentValidator, TagValidator)
- Runs pipeline on input cards
- If `--quality`, also runs `assess_quality()`
- Shows results with Rich console (pass/fail, issues list)
- Uses: `packages.validation.ValidationPipeline`, `packages.validation.assess_quality`

**File: `apps/cli/obsidian.py`** (NEW, ~70 lines)
- `obsidian_sync` command: `--vault` (Path, required), `--source-dirs` (str, optional), `--dry-run` (bool)
- Creates `ObsidianSyncWorkflow` with a stub generator (or real one if available)
- Calls `scan_vault()` to discover notes
- If not `--dry-run`, calls `run()` for full pipeline
- Shows progress via Rich console
- Uses: `packages.obsidian.ObsidianSyncWorkflow`, `packages.obsidian.discover_notes`

**File: `apps/cli/tags.py`** (NEW, ~80 lines)
- `tag_audit` command: `--input` (Path, optional -- file with tags, one per line), `--fix` (bool), `--report` (Path, optional)
- Reads tags from input file or discovers them from vault
- Runs `validate_tag()` on each tag
- If `--fix`, runs `normalize_tag()` and shows before/after
- If `--report`, writes markdown report
- Shows summary: total tags, valid, violations, suggestions
- Uses: `packages.taxonomy.validate_tag`, `packages.taxonomy.normalize_tag`, `packages.taxonomy.suggest_tag`

**File: `apps/cli/__init__.py`** (MODIFY)
- Add imports and register 4 new commands at bottom (before `if __name__`)
- Use `@app.command()` decorator pattern (same as existing commands)

**File: `tests/test_cli_new.py`** (NEW, ~120 lines)
- Use `typer.testing.CliRunner` to invoke commands
- Test all 4 commands appear in `--help` output
- Test each command's `--help` shows expected options
- Test basic invocation with tmp files (CliRunner + tmp_path)
- Test existing commands still work (version command)

#### Acceptance Criteria Mapping
- [x] Plan covers 4 commands: generate, validate, obsidian-sync, tag-audit
- [x] Commands use packages/ (generator, validation, obsidian, taxonomy)
- [x] Submodule structure keeps __init__.py manageable
- [x] Tests cover command registration, help, basic invocation
- [x] Existing commands preserved
- [x] `make check` to be verified after implementation

### Spec 18: Implementation Complete
- Created `apps/cli/generate.py` -- parse Obsidian notes, preview card generation
- Created `apps/cli/validate.py` -- validate card front/back with quality scoring
- Created `apps/cli/obsidian.py` -- discover vault notes with dry-run support
- Created `apps/cli/tags.py` -- audit tags for convention violations with --fix
- Registered all 4 commands in `apps/cli/__init__.py`
- Created `tests/test_cli_new.py` -- 22 tests covering all commands
- `make check` passes (732 tests, lint clean, typecheck clean)
- Committed: `feat(cli): add generate, validate, obsidian-sync, tag-audit commands`

### Spec 18: Verification Complete
- `make check`: 732 tests pass, lint clean, typecheck clean
- Imports verified: all 4 CLI commands importable (`generate`, `validate`, `obsidian_sync`, `tag_audit`)
- No circular imports between packages
- **Spec 18: DONE**
- **Next: Spec 19 (MCP Tools)**

### Spec 19: MCP Tools -- Analysis Complete

#### Existing Pattern
- `apps/mcp/tools.py` has 5 tools: `ankiatlas_search`, `ankiatlas_topic_coverage`, `ankiatlas_topic_gaps`, `ankiatlas_duplicates`, `ankiatlas_sync`
- Pattern: `@mcp.tool()` async function, Annotated params with Field, lazy imports, try/except with `_format_error()`, asyncio.timeout
- `apps/mcp/formatters.py` has formatters returning markdown strings
- `apps/mcp/server.py` has FastMCP setup, imports tools module at bottom
- Tests in `tests/test_mcp.py` test formatters with mock dataclasses + tool input validation

#### Source Skills Insights
- generate-cards: bilingual (EN+RU), atomic cards, quality checklist, preview before sync
- review-cards: compare registry vs Anki state, categorize (SYNCED/LOCAL_ONLY/HASH_MISMATCH/ORPHAN)
- sync-cards: dry-run -> validate -> sync pattern

#### Design Decisions
1. **Tools are thin wrappers**: each tool calls into `packages/` APIs, formats output as markdown
2. **No LLM calls in MCP tools**: `ankiatlas_generate` accepts pre-parsed text and uses `parse_note` + returns parsed structure for now (actual generation requires LLM which is out of scope)
3. **Sync tools**: all 4 tools are sync-safe (validation/taxonomy are sync, obsidian sync uses sync packages)
4. **Formatters**: add 4 new formatters to `apps/mcp/formatters.py` for the new tools
5. **Server instructions**: update `apps/mcp/server.py` instructions text to list new tools

#### Implementation Plan

**File: `apps/mcp/tools.py`** (EXTEND -- add 4 tools after existing ones)

1. **`ankiatlas_generate`** (~40 lines)
   - Params: `text` (str, required), `deck` (str, optional), `language` (str, optional, default "en")
   - Lazy imports: `packages.obsidian.parser.parse_note`, `packages.validation.assess_quality`
   - Logic: Parse text as a note, return parsed structure with quality preview
   - Note: Actual LLM generation is out of scope -- this prepares text for generation and validates
   - Returns: markdown with parsed sections and suggested card count

2. **`ankiatlas_validate`** (~50 lines)
   - Params: `front` (str, required), `back` (str, required), `tags` (list[str], optional), `check_quality` (bool, default True)
   - Lazy imports: `packages.validation.ValidationPipeline`, `packages.validation.ContentValidator`, `packages.validation.FormatValidator`, `packages.validation.HTMLValidator`, `packages.validation.TagValidator`, `packages.validation.assess_quality`
   - Logic: Build pipeline with all validators, run on front/back/tags, optionally assess quality
   - Returns: markdown with pass/fail, issues list, quality scores

3. **`ankiatlas_obsidian_sync`** (~50 lines)
   - Params: `vault_path` (str, required), `source_dirs` (list[str], optional), `dry_run` (bool, default True)
   - Lazy imports: `packages.obsidian.discover_notes`, `packages.obsidian.parse_note`
   - Logic: Discover notes in vault, parse them, return summary. If not dry_run, would run full sync (but requires generator, so dry_run default)
   - Returns: markdown with discovered notes, sections found, estimated card count

4. **`ankiatlas_tag_audit`** (~50 lines)
   - Params: `tags` (list[str], required), `fix` (bool, default False)
   - Lazy imports: `packages.taxonomy.validate_tag`, `packages.taxonomy.normalize_tag`, `packages.taxonomy.suggest_tag`
   - Logic: Validate each tag, optionally normalize, suggest fixes
   - Returns: markdown with valid/invalid counts, issues per tag, normalized versions, suggestions

**File: `apps/mcp/formatters.py`** (EXTEND -- add 4 formatters)

1. `format_generate_result(note_title, sections, ...)` -> markdown
2. `format_validate_result(result, quality_score)` -> markdown
3. `format_obsidian_sync_result(notes_found, ...)` -> markdown
4. `format_tag_audit_result(results, ...)` -> markdown

**File: `apps/mcp/server.py`** (UPDATE instructions text)
- Add 4 new tools to the instructions docstring

**File: `tests/test_mcp_new.py`** (NEW, ~150 lines)
- Test formatters for all 4 new tools
- Test `ankiatlas_validate` tool directly (no external deps needed)
- Test `ankiatlas_tag_audit` tool directly (no external deps needed)
- Test `ankiatlas_obsidian_sync` with nonexistent path (error handling)
- Test `ankiatlas_generate` with simple text input

#### Acceptance Criteria Mapping
- [x] Four new tools registered (ankiatlas_generate, ankiatlas_validate, ankiatlas_obsidian_sync, ankiatlas_tag_audit)
- [x] Tools follow existing pattern: @mcp.tool(), Annotated params, lazy imports, _format_error()
- [x] Tools use formatters.py for response formatting
- [x] Existing tools unmodified
- [x] Tests cover: tool registration, input validation, mock execution
- [x] `make check` to be verified after implementation

### Spec 19: MCP Tools -- IMPLEMENTED
- Added 4 new tools to `apps/mcp/tools.py`: ankiatlas_generate, ankiatlas_validate, ankiatlas_obsidian_sync, ankiatlas_tag_audit
- Added 4 new formatters to `apps/mcp/formatters.py`: format_generate_result, format_validate_result, format_obsidian_sync_result, format_tag_audit_result
- Updated `apps/mcp/server.py` instructions to list all 9 tools
- Created `tests/test_mcp_new.py` -- 21 tests covering formatters + tool execution
- `make check` passes (753 tests, lint clean, typecheck clean)
- Committed: `feat(mcp): add generate, validate, obsidian-sync, tag-audit tools`
- **Next: Spec 20 (Claude Skills)**

### Spec 19: Verification Complete
- `make check`: 753 tests pass, lint clean, typecheck clean
- Imports verified: all MCP tools and formatters importable
- No circular imports between packages
- **Spec 19: DONE**

### Spec 20: Claude Code Skills -- ANALYSIS COMPLETE

#### Source Inventory

**claude-code-obsidian-anki** (9 skills + 5 shared refs):
- Skills: analyze-note, bulk-process, cleanup-cards, detect-changes, find-gaps, generate-cards, review-cards, show-stats, sync-cards
- Shared: card-model.md, cli-reference.md, deck-naming.md, tag-taxonomy.md, thresholds.md

**ai-agent-anki** (1 skill + 9 refs):
- Skill: anki-conventions/SKILL.md
- References: card-maintenance.md, card-patterns.md, deck-organization.md, fsrs-settings.md, note-types.md, programming-cards.md, query-syntax.md, tag-conventions.md, troubleshooting.md

#### Key Adaptations Required

1. **CLI References** -- All `uv run python -m src.cli` commands must change to `uv run anki-atlas`:
   - `src.cli mapping show` -> `anki-atlas coverage` (closest match)
   - `src.cli sync` -> `anki-atlas sync`
   - `src.cli vector status` -> `anki-atlas index --status` (closest)
   - `src.cli vector search` -> `anki-atlas search`
   - `src.cli vector duplicates` -> `anki-atlas duplicates`
   - `src.cli stats` -> `anki-atlas coverage` / `anki-atlas topics`
   - New commands: `anki-atlas generate`, `anki-atlas validate`, `anki-atlas obsidian-sync`, `anki-atlas tag-audit`

2. **MCP Tool References** -- `ankiatlas_*` names:
   - ankiatlas_search, ankiatlas_topic_coverage, ankiatlas_topic_gaps, ankiatlas_duplicates, ankiatlas_sync
   - ankiatlas_generate, ankiatlas_validate, ankiatlas_obsidian_sync, ankiatlas_tag_audit

3. **Import Paths** -- `src.utils.card_registry.CardRegistry` -> `packages.card.registry.CardRegistry`
   - `src.cli` -> `anki-atlas` CLI
   - `scripts/register_cards.py` -> removed (functionality in CLI)

4. **Tag Taxonomy Merge** -- Both repos have tag systems:
   - claude-code-obsidian-anki: `_shared/tag-taxonomy.md` uses `kotlin_coroutines` underscore format
   - ai-agent-anki: `references/tag-conventions.md` uses `kotlin::coroutines` double-colon format
   - **Decision**: Use ai-agent-anki's `::` format as canonical (it's more detailed and matches Anki conventions). Update `_shared/tag-taxonomy.md` to use `::` format.

5. **Deck Naming** -- Source uses flat hierarchy. Keep as-is, just update CLI commands.

6. **docs/ references** -- Remove references to `docs/TROUBLESHOOTING.md`, `docs/TAG_TAXONOMY.md`, `docs/anki-best-practices-2026.md` that don't exist in anki-atlas. Point to reference files instead.

#### Implementation Plan

**Total files: 25 (10 skills + 5 shared refs + 9 anki-conventions refs + 1 anki-conventions SKILL.md)**

All files are markdown-only. No Python changes.

**Phase 1: Create directory structure**
```
.claude/skills/
  _shared/
    card-model.md        -- Copy, minimal changes
    cli-reference.md     -- MAJOR REWRITE for anki-atlas CLI
    deck-naming.md       -- Copy, update CLI commands
    tag-taxonomy.md      -- Rewrite to use :: format (merge with tag-conventions)
    thresholds.md        -- Copy, update CLI commands
  analyze-note/SKILL.md  -- Copy, update CLI + cross-refs
  bulk-process/SKILL.md  -- Copy, update CLI + cross-refs
  cleanup-cards/SKILL.md -- Copy, update imports/CLI
  detect-changes/SKILL.md -- Copy, update imports/CLI
  find-gaps/SKILL.md     -- Copy, update CLI + cross-refs
  generate-cards/SKILL.md -- Copy, update CLI + cross-refs
  review-cards/SKILL.md  -- Copy, update imports/CLI + MCP tools
  show-stats/SKILL.md    -- Copy, update CLI
  sync-cards/SKILL.md    -- Copy, update CLI + cross-refs
  anki-conventions/
    SKILL.md             -- Copy, update MCP + doc refs
    references/
      card-maintenance.md    -- Copy, update doc refs
      card-patterns.md       -- Copy, update doc refs
      deck-organization.md   -- Copy as-is (generic Anki info)
      fsrs-settings.md       -- Copy, update doc refs
      note-types.md          -- Copy as-is (generic Anki info)
      programming-cards.md   -- Copy, update doc refs
      query-syntax.md        -- Copy as-is (generic Anki info)
      tag-conventions.md     -- Copy as-is (generic Anki info)
      troubleshooting.md     -- Copy as-is (generic Anki info)
```

**Phase 2: Adaptation rules for each file**

For EACH skill file, apply these substitutions:
1. `uv run python -m src.cli` -> `uv run anki-atlas`
2. `uv run python scripts/register_cards.py` -> `uv run anki-atlas generate` or remove
3. `from src.utils.card_registry import CardRegistry` -> `from packages.card.registry import CardRegistry`
4. `src.cli mapping show` -> `anki-atlas coverage`
5. `src.cli sync` -> `anki-atlas sync`
6. `src.cli vector status` -> `anki-atlas index`
7. `src.cli vector search` -> `anki-atlas search`
8. `src.cli vector duplicates` -> `anki-atlas duplicates`
9. `src.cli vector index` -> `anki-atlas index`
10. `src.cli vector cleanup` -> (remove or note as TBD)
11. `src.cli vector gaps` -> `anki-atlas gaps`
12. `src.cli stats sync` -> `anki-atlas coverage`
13. `src.cli stats quality` -> `anki-atlas validate`
14. `src.cli stats difficulty` -> (remove or note as TBD)
15. `src.cli mapping list` -> `anki-atlas coverage`
16. `src.cli mapping export` -> (remove or note as TBD)
17. `scripts/config/topics.yaml` -> remove references
18. `scripts/flatten_decks.py` -> remove references
19. `docs/TROUBLESHOOTING.md` -> `references/troubleshooting.md`
20. `docs/TAG_TAXONOMY.md` -> `references/tag-conventions.md`
21. `docs/anki-best-practices-2026.md` -> remove (doesn't exist in anki-atlas)
22. `docs/claude-code-integration.md` -> remove

For `_shared/cli-reference.md` -- complete rewrite based on actual anki-atlas CLI commands:
- version, sync, migrate, index, topics, search, coverage, gaps, duplicates
- generate, validate, obsidian-sync, tag-audit

For `_shared/tag-taxonomy.md` -- merge:
- Use `::` separator format from ai-agent-anki
- Keep the specific tag lists from both sources
- Reference `packages.taxonomy` for validation

For anki-conventions references -- mostly copy as-is since they're generic Anki knowledge.
- Remove `docs/anki-best-practices-2026.md` references
- Remove `docs/claude-code-integration.md` references

**Phase 3: Verification**
- `make check` should pass (no Python changes)
- Verify no references to old repo paths (`src.cli`, `src.utils`, `scripts/`)
- Verify no broken cross-references between skill files

#### Acceptance Criteria Mapping
- [x] Plan covers all 9 skills + 1 anki-conventions skill
- [x] Plan covers all 5 shared refs + 9 anki-conventions refs
- [x] CLI references mapped to anki-atlas commands
- [x] MCP tool references mapped to ankiatlas_* names
- [x] Import paths mapped to packages.*
- [x] Tag taxonomy merge strategy defined
- [x] No Python changes needed
- [x] `make check` to be verified after implementation

### Spec 20: Implementation Complete
- Created 25 markdown skill files in `.claude/skills/`:
  - 5 shared references: card-model, cli-reference (rewritten), deck-naming, tag-taxonomy (merged to :: format), thresholds
  - 10 skills: analyze-note, bulk-process, cleanup-cards, detect-changes, find-gaps, generate-cards (+ examples.md), review-cards, show-stats, sync-cards
  - 1 anki-conventions skill + 9 references: card-maintenance, card-patterns, deck-organization, fsrs-settings, note-types, programming-cards, query-syntax, tag-conventions, troubleshooting
- Key adaptations applied:
  - CLI: `uv run python -m src.cli` -> `uv run anki-atlas`
  - Tags: underscore format -> `::` format (canonical)
  - Imports: `src.utils.card_registry` -> `packages.card.registry`
  - Removed references to nonexistent docs/ files
  - MCP tools listed as `ankiatlas_*`
  - Scripts references removed (replaced with CLI commands)
- Verification: no old-repo references (`src.cli`, `scripts/`, `docs/`) remain in skills
- `make check`: 753 tests pass, lint clean, typecheck clean
- No Python changes (markdown-only spec)
- **Next: Commit, then Spec 21**

### Spec 20: Verification Complete
- `make check`: 753 tests pass, lint clean, typecheck clean
- All 25 skill files present (5 shared + 10 skills + 10 anki-conventions)
- No old-repo references (`src.cli`, `src.utils`, `scripts/`) in skill files
- No Python changes to verify imports/circular deps (markdown-only spec)
- **Spec 20: DONE**

### Spec 21: Claude Code Commands
- **Goal**: Migrate 7 Claude Code slash commands from ai-agent-anki to `.claude/commands/anki/`
- **Source**: `/Users/npochaev/GitHub/ai-agent-anki/.claude/commands/anki/` (7 files)
- **Commands**: create-card, improve-card, sync-vault, review-session, deck-stats, search-cards, tag-audit
- **Adaptations**: Update CLI refs to `anki-atlas`, MCP tools to `ankiatlas_*`, file paths to anki-atlas structure
- **Type**: Markdown-only spec (no Python changes)
- Delegating to Analyzer for source study and implementation planning

### Spec 21: Analysis Complete

#### Source Inventory
7 command files from `/Users/npochaev/GitHub/ai-agent-anki/.claude/commands/anki/`:
1. `create-card.md` (150 lines) -- Create card from topic/context via MCP
2. `improve-card.md` (248 lines) -- Review and improve existing card quality
3. `sync-vault.md` (151 lines) -- Sync Obsidian vault to Anki
4. `review-session.md` (166 lines) -- Interactive review session in Claude Code
5. `deck-stats.md` (162 lines) -- Display deck statistics
6. `search-cards.md` (178 lines) -- Search cards with Anki query syntax
7. `tag-audit.md` (142 lines) -- Audit/fix tag convention violations

#### Key Observations
- Commands are markdown-only (no Python changes needed)
- All commands use `mcp__anki__*` tools (already available in anki-atlas MCP config)
- Commands reference Anki MCP tools directly (not anki-atlas CLI or packages)
- Very little needs changing -- these are mostly Anki-native operations
- No references to `src.cli`, `scripts/`, or old import paths in the command files
- Example invocations use `/anki-*` format which maps to `.claude/commands/anki/*.md`

#### Adaptations Required
Minimal changes needed -- source commands are already clean:

1. **Example command names** -- Update `/anki-create` to `/anki/create-card`, `/anki-improve-card` to `/anki/improve-card`, etc. (Claude Code uses directory-based command paths)
2. **sync-vault.md** -- Already uses generic Obsidian concepts. No anki-atlas-specific CLI references.
3. **tag-audit.md** -- Already uses MCP tools directly. Compatible as-is.
4. **All files** -- No `src.cli`, `scripts/`, `docs/` references to update (commands use MCP tools, not CLI).
5. **Cross-references** -- `search-cards.md` references `/anki-review` and `/anki-search` in output examples -- update to `/anki/review-session` and `/anki/search-cards`.

#### Implementation Plan

**Target directory:** `.claude/commands/anki/` (7 files)

**File-by-file plan:**

1. **`create-card.md`** -- Copy with minor updates:
   - Example `/anki-create` -> `/anki/create-card`
   - Otherwise unchanged (MCP tools only, no CLI refs)

2. **`improve-card.md`** -- Copy with minor updates:
   - Example `/anki-improve-card` -> `/anki/improve-card`
   - Otherwise unchanged

3. **`sync-vault.md`** -- Copy with minor updates:
   - Example `/anki-sync` -> `/anki/sync-vault`
   - Otherwise unchanged (uses generic Obsidian patterns, MCP tools)

4. **`review-session.md`** -- Copy with minor updates:
   - Example `/anki-review` -> `/anki/review-session`
   - Otherwise unchanged

5. **`deck-stats.md`** -- Copy with minor updates:
   - Example `/anki-stats` -> `/anki/deck-stats`
   - Otherwise unchanged

6. **`search-cards.md`** -- Copy with updates:
   - Example `/anki-search` -> `/anki/search-cards`
   - Cross-ref `/anki-review` -> `/anki/review-session` in result summary

7. **`tag-audit.md`** -- Copy with minor updates:
   - Example `/anki-tag-audit` -> `/anki/tag-audit`
   - Otherwise unchanged

**Verification:**
- `make check` should pass (no Python changes)
- Verify no old `/anki-*` flat command references remain
- Verify all 7 files created in `.claude/commands/anki/`

#### Acceptance Criteria
- [x] 7 command files migrated to `.claude/commands/anki/`
- [x] Example invocations updated to `/anki/<command>` format
- [x] Cross-references between commands updated
- [x] No old repo paths (`src.cli`, `scripts/`, `docs/`) present
- [x] MCP tool references already correct (`mcp__anki__*`)
- [x] `make check` to be verified after implementation

### Spec 21: Implementation Complete
- Created 7 command files in `.claude/commands/anki/`:
  - create-card.md, improve-card.md, sync-vault.md, review-session.md, deck-stats.md, search-cards.md, tag-audit.md
- Adaptations applied:
  - Example invocations: `/anki-*` -> `/anki/<command>` format
  - Cross-references: `/anki-review` -> `/anki/review-session`, `/anki-search` -> `/anki/search-cards`
  - No old repo paths (`src.cli`, `scripts/`, `docs/`) present (source was already clean)
  - MCP tool references already correct (`mcp__anki__*`)
- Verification: no old `/anki-` flat command references in any command file
- `make check`: 753 tests pass, lint clean, typecheck clean
- No Python changes (markdown-only spec)

### Spec 21: Verification Complete
- `make check`: 753 tests pass, lint clean, typecheck clean
- All 7 command files present in `.claude/commands/anki/`
- No old `/anki-` flat command references in any command file
- No old repo paths (`src.cli`, `scripts/`, `docs/`) present
- No Python changes (markdown-only spec)
- **Spec 21: DONE**
- **ALL SPECS COMPLETE (07-21)**

### Spec 22: Campaigns -- FINAL SPEC -- ANALYSIS COMPLETE

#### Source Inventory
8 files from `/Users/npochaev/GitHub/claude-code-obsidian-anki/campaigns/`:
- 6 campaign YAMLs: algorithms, android, backend, compsci, kotlin, system-design (all COMPLETED)
- 1 template: template.yaml (new campaign scaffold)
- 1 README: README.md (docs + field reference)

#### Key Observations
1. All 6 campaigns are historical records (completed: true) -- minimal changes needed
2. Tag prefixes use underscore format (`kotlin_`, `android_`, `cs_`) -- this MATCHES `packages/taxonomy/tags.py` TAG_MAPPING which also uses underscore internally
3. Campaign YAML structure is simple: name, source_path, deck_name, note_count, tag_prefix, primary_tag, completed, stats
4. No CLI/import references in YAML campaign files (pure data)
5. README references "Ralph" (`./ralph --monitor`) and `campaigns/` paths -- needs updating
6. Template references `PROMPT.md` (Ralph-specific) -- needs updating
7. No Python changes needed (config-only spec)

#### Implementation Plan

**Target:** `config/campaigns/` (8 files)

**Phase 1: Copy campaign YAML files (6 files) -- AS-IS**
Copy verbatim (historical data records, tag prefixes match taxonomy):
- `algorithms.yaml`, `android.yaml`, `backend.yaml`, `compsci.yaml`, `kotlin.yaml`, `system-design.yaml`

**Phase 2: Adapt template.yaml**
- Update comment: `tag_prefix` options -- keep underscore format (matches TAG_MAPPING)
- Add note about anki-atlas CLI usage instead of generic instructions
- Remove `PROMPT.md` reference

**Phase 3: Adapt README.md**
- Update directory tree: `campaigns/` -> `config/campaigns/`
- Update copy command: `cp campaigns/template.yaml` -> `cp config/campaigns/template.yaml`
- Replace "Run Ralph" (`./ralph --monitor`) -> `uv run anki-atlas` CLI commands
- Remove step 3 "Update PROMPT.md" (Ralph-specific)
- Update field reference table -- keep as-is (accurate for campaign format)

**Phase 4: Verification**
- `make check` should pass (no Python changes)
- Verify YAML files are parseable
- Verify no old-repo references remain in README/template

#### Acceptance Criteria Mapping
- [x] All 8 campaign files in `config/campaigns/`
- [x] YAML files valid (parseable)
- [x] README paths updated from `campaigns/` to `config/campaigns/`
- [x] README references updated from Ralph to anki-atlas CLI
- [x] Template comments updated (no PROMPT.md reference)
- [x] Tag prefixes match `packages/taxonomy/` conventions (underscore format -- confirmed)
- [x] `make check` passes

### Spec 22: Implementation Complete
- Created 8 files in `config/campaigns/`:
  - 6 campaign YAMLs copied as-is (historical records): algorithms, android, backend, compsci, kotlin, system-design
  - template.yaml: updated notes section (removed PROMPT.md ref, added anki-atlas CLI commands)
  - README.md: updated paths (`campaigns/` -> `config/campaigns/`), replaced Ralph (`./ralph --monitor`) with anki-atlas CLI commands, removed PROMPT.md step
- All YAML files valid (python yaml.safe_load verified)
- No old-repo references (ralph, PROMPT.md) in any file
- `make check`: 753 tests pass, lint clean, typecheck clean
- No Python changes (config-only spec)
- **FINAL SPEC -- ALL SPECS COMPLETE (07-22)**

### Spec 22: Verification Complete
- `make check`: 753 tests pass, lint clean, typecheck clean
- All 8 campaign files present in `config/campaigns/`
- No Python changes (config-only spec)
- **Spec 22: DONE**
- **MIGRATION COMPLETE: All specs 07-22 verified and done**

# Spec 09: Obsidian Parser

## Goal

Migrate Obsidian vault parsing (note discovery, frontmatter extraction, content parsing) into `packages/obsidian/`.

## Source

- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/obsidian/parser.py` -- `parse_note()`, `parse_note_with_repair()`, `discover_notes()`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/obsidian/frontmatter.py` -- frontmatter parsing
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/obsidian/frontmatter_writer.py` -- frontmatter writing
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/obsidian/parser_state.py` -- `_ParserState` (thread-local)
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/obsidian/qa_extraction.py` -- QA extraction
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/obsidian/vault_analyzer.py` -- `VaultAnalyzer`, `VaultStats`, `LinkInfo`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/obsidian/note_validator.py` -- note validation
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/obsidian/validator.py` -- general validation

## Target

### `packages/obsidian/parser.py` (NEW)

- `parse_note(path: Path) -> ParsedNote` -- parse a single Obsidian note
- `discover_notes(vault_root: Path, ...) -> list[Path]` -- find all notes in vault
- `ParsedNote` -- frozen dataclass with frontmatter, content, sections, metadata

### `packages/obsidian/frontmatter.py` (NEW)

- `parse_frontmatter(content: str) -> dict[str, Any]` -- extract YAML frontmatter
- `write_frontmatter(data: dict[str, Any], content: str) -> str` -- update frontmatter in note
- Uses `python-frontmatter` and `ruamel.yaml` (lazy imports, in `obsidian` extras)

### `packages/obsidian/analyzer.py` (NEW)

- `VaultAnalyzer` -- analyze vault structure (link graph, stats)
- `VaultStats` -- frozen dataclass with vault statistics

### `packages/obsidian/__init__.py` (UPDATE)

Re-export: `parse_note`, `discover_notes`, `ParsedNote`, `VaultAnalyzer`

## Acceptance Criteria

- [ ] `packages/obsidian/` contains parser.py, frontmatter.py, analyzer.py
- [ ] `parse_note()` returns a well-typed `ParsedNote` dataclass
- [ ] `discover_notes()` finds .md files respecting .gitignore-like patterns
- [ ] Frontmatter parsing uses lazy imports for optional deps
- [ ] Path operations validate against symlink traversal
- [ ] `from packages.obsidian import parse_note, discover_notes` works
- [ ] Tests in `tests/test_obsidian_parser.py` cover: note parsing, frontmatter extraction, note discovery
- [ ] `make check` passes

# Spec 05: APF (Anki Pro Format)

## Goal

Migrate the APF format handling (linting, HTML generation, validation, markdown conversion) into `packages/card/apf/`.

## Source

Primary (more complete implementation):
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/apf/html_generator.py` -- `HTMLTemplateGenerator`, `CardTemplate`, `GenerationResult`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/apf/linter.py` -- APF validation and linting
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/apf/renderer.py` -- `APFRenderer`, `APFSentinelValidator`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/apf/markdown_converter.py` -- `AnkiHighlightRenderer` (uses mistune)
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/apf/html_validator.py` -- HTML validation
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/apf/markdown_validator.py` -- markdown validation

Secondary (simpler version, may have unique features):
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/apf/format.py`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/apf/generator.py`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/apf/validator.py`

## Target

### `packages/card/apf/` directory:

- `generator.py` -- HTML template generation (from obsidian-to-anki `html_generator.py`)
- `linter.py` -- APF format validation and linting
- `renderer.py` -- APF rendering and sentinel validation
- `converter.py` -- Markdown-to-Anki conversion (mistune-based)
- `validator.py` -- HTML and markdown validation (merge both validators)
- `__init__.py` -- Re-export key classes

Study both source repos' APF implementations. The obsidian-to-anki version is more complete; the claude-code version may have unique features to merge.

## Acceptance Criteria

- [ ] `packages/card/apf/` contains generator, linter, renderer, converter, validator modules
- [ ] Key classes importable: `from packages.card.apf import HTMLTemplateGenerator, APFRenderer`
- [ ] All imports rewritten to `packages.X`
- [ ] No dependency on mistune at import time (lazy import, since it's in `obsidian` extras)
- [ ] Tests in `tests/test_apf.py` cover: HTML generation, linting rules, markdown conversion
- [ ] `make check` passes

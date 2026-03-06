# Spec 11: Prompts System

## Goal

Migrate LLM prompt templates for card generation into `packages/generator/prompts/`.

## Source

obsidian-to-anki prompts:
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/prompts/template_parser.py` -- template parsing/rendering

claude-code-obsidian-anki prompts (8 modules):
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/prompts/__init__.py`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/prompts/card_generation.py`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/prompts/card_splitting.py`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/prompts/context_enrichment.py`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/prompts/enhancement.py`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/prompts/memorization_assessment.py`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/prompts/post_validation.py`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/prompts/pre_validation.py`
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/prompts/template_parser.py`

## Target

### `packages/generator/prompts/` directory:

- `templates.py` -- prompt template rendering engine (from obsidian-to-anki `template_parser.py`)
- `generation.py` -- card generation prompts (system + user prompts)
- `validation.py` -- pre/post validation prompts
- `enhancement.py` -- card enhancement and splitting prompts
- `__init__.py` -- re-export key functions

Study both sources. The claude-code version has more granular prompt modules; the obsidian-to-anki version has a template parser. Unify:
- Template rendering from obsidian-to-anki
- Prompt content organized by stage (generation, validation, enhancement)
- Prompts as string constants or template functions, not files

## Acceptance Criteria

- [ ] `packages/generator/prompts/` contains templates.py, generation.py, validation.py, enhancement.py
- [ ] Prompt functions accept parameters and return formatted prompt strings
- [ ] No hardcoded file paths for prompt templates
- [ ] `from packages.generator.prompts import generation` works
- [ ] Tests in `tests/test_prompts.py` cover: template rendering, prompt formatting with parameters
- [ ] `make check` passes

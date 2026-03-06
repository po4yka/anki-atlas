# Spec 12: Validation Pipeline

## Goal

Migrate card validation orchestration into `packages/validation/`.

## Source

obsidian-to-anki validation:
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/validation/orchestrator.py` -- `NoteValidator` (main orchestrator)
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/validation/base.py` -- `AutoFix`, `Severity` enum
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/validation/content_validator.py` -- `ContentValidator`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/validation/format_validator.py` -- `FormatValidator`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/validation/html_validator.py` -- `HTMLValidator`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/validation/yaml_validator.py` -- `YAMLValidator`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/validation/android_validator.py` -- `AndroidValidator`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/validation/link_validator.py` -- `LinkValidator`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/validation/parallel_validator.py` -- `ParallelValidator`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/validation/ai_fixer.py` -- AI-powered auto-fix

ai-agent-anki quality rubric:
- `/Users/npochaev/GitHub/ai-agent-anki/.ralph/specs/card-review.md` -- 5-dimension quality rubric, review decision tree

claude-code-obsidian-anki:
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/src/validation/card_quality.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/agents/pre_validator.py`

## Target

### `packages/validation/pipeline.py` (NEW)

- `ValidationPipeline` -- orchestrator that runs validators in sequence
- `ValidationResult` -- frozen dataclass with issues, auto-fixes, severity
- `ValidationIssue` -- single issue with severity, message, location
- `Severity` -- enum (ERROR, WARNING, INFO)

### `packages/validation/validators.py` (NEW)

Individual validator implementations:
- `ContentValidator` -- checks card content quality
- `FormatValidator` -- checks APF format compliance
- `HTMLValidator` -- validates HTML in card fields
- `TagValidator` -- validates tags against taxonomy

### `packages/validation/quality.py` (NEW)

Quality scoring based on ai-agent-anki's 5-dimension rubric:
- `QualityScore` -- frozen dataclass with dimension scores
- `assess_quality(card) -> QualityScore` -- score a card
- Dimensions: clarity, atomicity, testability, memorability, accuracy

### `packages/validation/__init__.py` (UPDATE)

Re-export: `ValidationPipeline`, `ValidationResult`, `Severity`, `QualityScore`

## Acceptance Criteria

- [ ] `packages/validation/` contains pipeline.py, validators.py, quality.py
- [ ] `ValidationPipeline` can chain multiple validators
- [ ] `QualityScore` implements the 5-dimension rubric
- [ ] `from packages.validation import ValidationPipeline, QualityScore` works
- [ ] Tests in `tests/test_validation.py` cover: pipeline execution, individual validators, quality scoring
- [ ] `make check` passes

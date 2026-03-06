# Migration Rules

## Source Study

- Read and understand source modules before migrating
- Map dependencies: what does this module import? what imports it?
- Identify the essential logic vs. boilerplate

## Adaptation

- Adapt to anki-atlas conventions -- do not copy verbatim
- Rewrite all imports:
  - `obsidian_anki_sync.X` -> `packages.X`
  - `src.X` -> `packages.X`
  - `from src.domain.entities.card import Card` -> `from packages.card.models import Card`
- Replace logging: `get_logger()` / `logging.getLogger()` -> `structlog.get_logger()`
- Replace print() -> structlog

## Quality Gates

- Every migrated module needs at least one test
- `make check` must pass after every spec
- No circular dependencies between packages
- New packages must have `__init__.py` + `py.typed`

## Commit Convention

- One commit per spec
- Format: `feat(<package>): migrate <component> from <source>`
- Examples:
  - `feat(card): migrate card domain models from claude-code-obsidian-anki`
  - `feat(llm): migrate provider abstraction from obsidian-to-anki`
  - `feat(taxonomy): migrate tag system from ai-agent-anki`

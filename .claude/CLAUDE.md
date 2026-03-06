# anki-atlas

Unified Anki flashcard platform: hybrid search index + card generation + obsidian sync + MCP tools.

## Commands

- `make check` -- Run all checks (lint + typecheck + test). **Gate for every commit.**
- `make test` -- Run pytest
- `make lint` -- Run ruff linter
- `make typecheck` -- Run mypy strict
- `make format` -- Format code with ruff

## Architecture

```
apps/       -- Application entry points (api, cli, mcp, worker)
packages/   -- Core library packages (shared, reusable)
tests/      -- All tests
config/     -- Configuration files
```

**Dependency rule:** `apps/` depends on `packages/`, `packages/` never depends on `apps/`.
Packages may depend on other packages but **no circular dependencies**.

## Conventions

- Python 3.13+
- `from __future__ import annotations` in every `.py` file
- mypy strict mode
- ruff formatter + linter (100 char line limit, double quotes)
- structlog for logging (no `print()`)
- Immutability-first: frozen dataclasses, `Final` where appropriate
- Complete type hints on all public APIs
- Max 600 lines per file

## Import Style

```python
from packages.common.config import Settings
from packages.anki.models import AnkiNote
from apps.cli import app
```

## Testing

- pytest with pytest-asyncio (asyncio_mode = "auto")
- Every module needs at least one test
- Tests in `tests/` mirror the package structure
- Run single tests: `uv run pytest tests/test_foo.py -x`

## Entry Points

- CLI: `uv run anki-atlas`
- MCP: `uv run anki-atlas-mcp`
- API: `uv run uvicorn apps.api.main:app`

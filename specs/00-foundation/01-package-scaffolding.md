# Spec 01: Package Scaffolding

## Goal

Create empty package directories for all new packages that will receive migrated code.

## Source

N/A -- this is greenfield scaffolding.

## Target

Create the following new packages under `packages/`:

```
packages/card/
  __init__.py
  py.typed
  apf/
    __init__.py

packages/generator/
  __init__.py
  py.typed
  prompts/
    __init__.py
  agents/
    __init__.py
  learning/
    __init__.py

packages/obsidian/
  __init__.py
  py.typed

packages/llm/
  __init__.py
  py.typed

packages/validation/
  __init__.py
  py.typed

packages/rag/
  __init__.py
  py.typed

packages/taxonomy/
  __init__.py
  py.typed
```

Each `__init__.py` should contain:
```python
from __future__ import annotations
```

Each `py.typed` should be an empty marker file.

## Acceptance Criteria

- [ ] All 7 new package directories exist under `packages/`
- [ ] Each has `__init__.py` with `from __future__ import annotations`
- [ ] Each has `py.typed` marker file
- [ ] Sub-packages (`card/apf/`, `generator/prompts/`, `generator/agents/`, `generator/learning/`) exist with `__init__.py`
- [ ] All packages are importable: `python -c "import packages.card; import packages.generator; import packages.obsidian; import packages.llm; import packages.validation; import packages.rag; import packages.taxonomy"`
- [ ] `make check` passes

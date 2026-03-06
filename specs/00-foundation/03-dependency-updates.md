# Spec 03: Dependency Updates

## Goal

Add new optional dependency groups to `pyproject.toml` for migrated code, and update mypy overrides.

## Source

- `/Users/npochaev/GitHub/obsidian-to-anki/pyproject.toml` -- dependency versions
- `/Users/npochaev/GitHub/claude-code-obsidian-anki/pyproject.toml` -- dependency versions

## Target

### `pyproject.toml` (MODIFY)

Add new optional dependency groups (preserve all existing groups):

```toml
[project.optional-dependencies]
# ... existing groups unchanged (dev, embeddings-openai, embeddings-google, embeddings-local) ...

llm = [
    "pydantic-ai-slim>=1.27.0",
    "langsmith>=0.4.56",
]
obsidian = [
    "python-frontmatter>=1.1.0",
    "mistune>=3.1.4",
    "ruamel.yaml>=0.18.16",
]
rag = [
    "chromadb>=1.3.5",
]
providers = [
    "openai>=2.9.0",
]
card = [
    "nh3>=0.3.2",
    "genanki>=0.13.1",
    "beautifulsoup4>=4.14.3",
]
all = [
    "anki-atlas[dev,embeddings-openai,embeddings-local,llm,obsidian,rag,providers,card]",
]
```

Add mypy overrides for new dependencies:

```toml
[[tool.mypy.overrides]]
module = ["frontmatter.*", "mistune.*", "nh3.*", "bs4.*", "chromadb.*", "genanki.*", "pydantic_ai.*", "langsmith.*", "ruamel.*"]
ignore_missing_imports = true
```

### Post-update

Run `uv lock` to regenerate the lock file.

## Acceptance Criteria

- [ ] New optional groups added to `pyproject.toml`: `llm`, `obsidian`, `rag`, `providers`, `card`, `all`
- [ ] mypy overrides include new third-party modules
- [ ] `uv lock` succeeds without errors
- [ ] `uv sync --all-extras` succeeds (or `uv sync --extra dev`)
- [ ] `make check` passes

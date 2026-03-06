# Ralph Migration Directive: Consolidate into anki-atlas

## Mission

Consolidate three Anki flashcard repositories into `anki-atlas` as the single unified repo:

1. **obsidian-to-anki** -- LLM pipeline for card generation (LangGraph, PydanticAI)
2. **claude-code-obsidian-anki** -- Claude Code card crafting with campaigns
3. **ai-agent-anki** -- Pure Claude Code config for card management (skills, commands, Ralph specs)

## Source Repositories (absolute paths)

| Repo | Path | Import Base |
|------|------|-------------|
| obsidian-to-anki | `/Users/npochaev/GitHub/obsidian-to-anki/` | `obsidian_anki_sync` |
| claude-code-obsidian-anki | `/Users/npochaev/GitHub/claude-code-obsidian-anki/` | `src` |
| ai-agent-anki | `/Users/npochaev/GitHub/ai-agent-anki/` | N/A (config-only) |

## Architecture Mapping

Where migrated code lands in anki-atlas's monorepo:

```
packages/
  common/          # (exists) + new types, extended exceptions
  anki/            # (exists) + connect.py (AnkiConnect client), sync/ extension
  indexer/         # (exists) unchanged
  search/          # (exists) unchanged
  analytics/       # (exists) unchanged
  jobs/            # (exists) unchanged
  card/            # NEW: card domain, APF format, registry
    models.py        <- claude-code domain/entities/card.py
    apf/             <- obsidian-to-anki apf/ + claude-code apf/
    registry.py      <- claude-code utils/card_registry.py
  generator/       # NEW: LLM card generation pipeline
    prompts/         <- obsidian-to-anki prompts/ + claude-code prompts/
    agents/          <- obsidian-to-anki agents/ (PydanticAI)
    learning/        <- obsidian-to-anki agents/agent_learning.py
  obsidian/        # NEW: vault parsing
    parser.py        <- obsidian-to-anki obsidian/parser.py
    frontmatter.py   <- obsidian-to-anki obsidian/frontmatter.py
    sync.py          <- obsidian-to-anki sync/note_scanner + processor
  llm/             # NEW: provider abstraction
    base.py          <- obsidian-to-anki providers/ (BaseLLMProvider)
    factory.py       <- obsidian-to-anki providers/ (ProviderFactory)
    ollama.py        <- obsidian-to-anki providers/ollama.py
    openrouter.py    <- obsidian-to-anki providers/openrouter/
  validation/      # NEW: card quality pipeline
    pipeline.py      <- obsidian-to-anki validation/orchestrator.py
    quality.py       <- ai-agent-anki 5-dimension rubric
  rag/             # NEW: retrieval-augmented generation
    indexer.py       <- obsidian-to-anki rag/indexer.py
    retriever.py     <- obsidian-to-anki rag/retriever.py
  taxonomy/        # NEW: tag system
    tags.py          <- ai-agent-anki TAGS.md + claude-code tag_taxonomy.py
    normalize.py     <- tag normalization and validation

apps/
  api/             # (exists) unchanged initially
  cli/             # (exists) + new commands: generate, validate, obsidian-sync, tag-audit
  mcp/             # (exists) + new tools: generate, validate, obsidian_sync
  worker.py        # (exists) unchanged
```

## Working Rules

1. **Study source thoroughly** before migrating -- understand what the code does, not just its shape
2. **Adapt to anki-atlas conventions** -- don't copy verbatim:
   - `from __future__ import annotations` in every file
   - Double quotes, 100 char line limit, ruff formatting
   - Immutability-first (frozen dataclasses, Final where appropriate)
   - structlog for logging (no print())
   - Complete type hints, mypy strict
3. **Rewrite imports**: `obsidian_anki_sync.X` -> `packages.X`, `src.X` -> `packages.X`
4. **Every migrated module needs at least one test**
5. **No circular dependencies** between packages
6. **`make check` must pass** after every spec (lint + typecheck + test)
7. **One commit per spec**: `feat(<package>): migrate <component> from <source>`
8. **Max 600 lines per file** -- split if necessary

## Spec Execution Order

Read specs from `specs/` in filename order (00-foundation first, then 01-domain, etc.).

For each spec:
1. Read the spec file completely
2. Study all source files referenced in the spec
3. Plan the implementation (write to SCRATCHPAD.md)
4. Implement: create/modify files in anki-atlas
5. Add tests
6. Run `make check` -- must pass
7. Commit with conventional commit message
8. Move to next spec

## Completion

When ALL specs are done and `make check` is green, emit `LOOP_COMPLETE`.

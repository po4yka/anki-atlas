# Spec 13: Generator Agents

## Goal

Migrate the LLM-based card generation agents into `packages/generator/agents/`.

## Source

PydanticAI agents (primary):
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/agents/pydantic_ai/generator_agent.py` -- `GeneratorAgentAI`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/agents/pydantic_ai/enhancement_agents.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/agents/pydantic_ai/pre_validator_agent.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/agents/pydantic_ai/post_validator_agent.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/agents/pydantic_ai/deps.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/agents/pydantic_ai/outputs.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/agents/pydantic_ai/streaming.py`

Legacy/models:
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/agents/generator.py` -- `GeneratorAgent` (legacy orchestrator)
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/agents/models.py` -- `GeneratedCard`, `GenerationResult`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/agents/exceptions.py` -- `GenerationError`, `StructuredOutputError`

## Target

### `packages/generator/agents/` directory:

- `models.py` -- `GeneratedCard`, `GenerationResult`, `GenerationDeps` (frozen dataclasses)
- `generator.py` -- Main generation agent (PydanticAI-based)
- `validator.py` -- Pre/post validation agents
- `enhancer.py` -- Enhancement agents (card improvement, splitting)
- `__init__.py` -- Re-export key classes

Focus on the PydanticAI implementation (modern, well-structured). Skip the legacy `GeneratorAgent` and LangGraph orchestrator -- they can be added later if needed.

Key adaptation:
- Import `pydantic-ai-slim` (in `llm` extras group)
- Use `packages.llm` for provider abstraction
- Use `packages.generator.prompts` for prompt templates
- Use `packages.validation` for quality checks

## Acceptance Criteria

- [ ] `packages/generator/agents/` contains models.py, generator.py, validator.py, enhancer.py
- [ ] `GeneratedCard` and `GenerationResult` are well-typed frozen dataclasses
- [ ] Generator agent uses PydanticAI for structured output
- [ ] Lazy import of `pydantic_ai` (optional dependency)
- [ ] `from packages.generator.agents import GeneratedCard, GenerationResult` works
- [ ] Tests in `tests/test_generator_agents.py` cover: model creation, generation flow (mock LLM)
- [ ] `make check` passes

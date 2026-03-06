# Spec 10: LLM Providers

## Goal

Migrate the LLM provider abstraction (base class, factory, concrete providers) into `packages/llm/`.

## Source

- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/providers/base.py` -- `BaseLLMProvider` (ABC), `LLMResponse`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/providers/factory.py` -- `ProviderFactory.create_provider()`, `PROVIDER_MAP`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/providers/ollama.py` -- `OllamaProvider`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/providers/lm_studio.py` -- `LMStudioProvider`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/providers/openrouter/provider.py` -- `OpenRouterProvider`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/providers/openrouter/models.py` -- `MODEL_CONTEXT_WINDOWS`, model constants
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/providers/openrouter/payload_builder.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/providers/openrouter/error_handler.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/providers/openrouter/retry_handler.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/providers/openrouter/token_calculator.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/providers/openrouter/streaming.py`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/providers/safety.py` -- `OllamaSafetyWrapper`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/providers/config_models.py`

## Target

### `packages/llm/base.py` (NEW)

- `BaseLLMProvider` -- ABC with `generate()` and `generate_async()` methods
- `LLMResponse` -- frozen dataclass with response text, token usage, model info
- `ProviderConfig` -- base configuration dataclass

### `packages/llm/factory.py` (NEW)

- `ProviderFactory` -- create provider instances by type string
- `ProviderType` -- enum of supported providers

### `packages/llm/ollama.py` (NEW)

- `OllamaProvider(BaseLLMProvider)` -- Ollama API client

### `packages/llm/openrouter.py` (NEW)

- `OpenRouterProvider(BaseLLMProvider)` -- OpenRouter API client
- Consolidate the openrouter/ sub-package into a single module (if under 600 lines), otherwise keep as sub-package

### `packages/llm/__init__.py` (UPDATE)

Re-export: `BaseLLMProvider`, `LLMResponse`, `ProviderFactory`, `OllamaProvider`, `OpenRouterProvider`

Note: LM Studio provider can be omitted initially -- it's a thin wrapper around OpenAI-compatible API.

## Acceptance Criteria

- [ ] `packages/llm/` contains base.py, factory.py, ollama.py, openrouter.py
- [ ] `BaseLLMProvider` ABC defines clear contract (generate, generate_async)
- [ ] `ProviderFactory.create_provider("ollama", ...)` returns `OllamaProvider`
- [ ] Uses httpx for HTTP, structlog for logging
- [ ] Uses `ProviderError` from `packages.common.exceptions`
- [ ] `from packages.llm import BaseLLMProvider, ProviderFactory` works
- [ ] Tests in `tests/test_llm_providers.py` cover: factory creation, provider interface, mock responses
- [ ] `make check` passes

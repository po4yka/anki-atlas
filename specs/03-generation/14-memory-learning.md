# Spec 14: Memory and Learning

## Goal

Migrate the agent learning/memory system for improving card generation over time.

## Source

- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/agents/` -- Look for files related to: `agent_learning.py`, `advanced_memory.py`, memory/learning patterns
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/infrastructure/cache/cache_manager.py` -- `CacheManager`
- `/Users/npochaev/GitHub/obsidian-to-anki/src/obsidian_anki_sync/infrastructure/cache/cache_strategy.py`

Study the source to understand:
1. How the system learns from past generation successes/failures
2. How few-shot examples are stored and retrieved
3. How generation quality improves over time

## Target

### `packages/generator/learning/` directory:

- `memory.py` -- Agent memory system for storing generation outcomes
- `examples.py` -- Few-shot example management (store, retrieve, rank)
- `feedback.py` -- Feedback loop for quality improvement
- `__init__.py` -- Re-export key classes

### Key abstractions:

- `GenerationMemory` -- stores past generation outcomes with quality scores
- `FewShotStore` -- manages curated examples for prompting
- `FeedbackCollector` -- records what worked/didn't for future improvement

If the source memory system is tightly coupled to LangGraph state, simplify to a clean interface that can be backed by SQLite or in-memory storage.

## Acceptance Criteria

- [ ] `packages/generator/learning/` contains memory.py, examples.py, feedback.py
- [ ] Clean interfaces not coupled to specific LLM framework
- [ ] `from packages.generator.learning import GenerationMemory, FewShotStore` works
- [ ] Tests in `tests/test_learning.py` cover: memory storage/retrieval, example ranking
- [ ] `make check` passes

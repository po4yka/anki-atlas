# Design: Full Rust Rewrite of anki-atlas

**Date:** 2026-03-08
**Status:** Approved
**Approach:** In-place replacement on feature branch, TDD via Ralph Loop

## Summary

Rewrite the entire anki-atlas Python project (~17.4K lines) to Rust, maintaining full feature parity. The rewrite uses axum + tokio for the async runtime, sqlx for PostgreSQL, and a custom Ralph Loop TDD hat collection to enforce red-green-refactor discipline.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Async runtime | tokio | Industry standard, axum requires it |
| Web framework | axum | Tower middleware ecosystem, type-safe extractors |
| CLI | clap (derive) | Most popular, good error messages |
| PostgreSQL | sqlx | Compile-time query checking |
| SQLite (Anki) | rusqlite | Mature, full SQLite API |
| Vector DB | qdrant-client | Official Rust SDK |
| HTTP client | reqwest | Built on hyper, ergonomic API |
| Serialization | serde | Universal standard |
| Logging | tracing | Async-aware, structured, spans |
| Errors | thiserror + anyhow | thiserror for libs, anyhow for bins |
| MCP | rmcp | Official Rust MCP SDK |
| Redis | rustis | Async, full Redis API |
| Config | figment | Composable, env var support |
| Mocking | mockall | Auto-generates mock impls from traits |

## Project Structure

```
Cargo.toml                    # workspace root
crates/
  common/                     # types, config, errors, tracing
  taxonomy/                   # tag normalization and validation
  database/                   # PostgreSQL pool, migrations
  anki-reader/                # SQLite reader, models, normalizer
  anki-sync/                  # sync engine, state, recovery
  indexer/                    # embeddings, qdrant, index service
  search/                     # FTS, fusion, reranker, service
  analytics/                  # taxonomy, coverage, duplicates
  card/                       # models, slug, registry, APF
  generator/                  # LLM-powered card generation
  llm/                        # provider abstraction
  obsidian/                   # vault parser, analyzer
  rag/                        # chunker, vector store
  validation/                 # pipeline, validators, quality
  jobs/                       # background job queue
bins/
  cli/                        # clap binary
  api/                        # axum binary
  mcp/                        # rmcp binary
  worker/                     # tokio job worker
```

**Dependency rule:** `bins/` -> `crates/`, never reverse. No circular deps between crates.

## Ralph Loop Strategy

Custom `tdd-rewrite.yml` with 4 hats enforcing red-green-refactor:

1. **Test Architect** (red) - writes failing tests from spec
2. **Implementer** (green) - minimum code to pass tests
3. **Refactorer** - cleans up, keeps tests green
4. **Verifier** - cargo test + clippy, checks acceptance criteria

Completion promise: `CRATE_COMPLETE`

### Execution Order (19 runs)

| Phase | Crates | Dependencies |
|-------|--------|-------------|
| 1. Foundation | common, taxonomy | none |
| 2. Data layer | database, anki-reader | common |
| 3. Core services | anki-sync, indexer, search | database, anki-reader |
| 4. Domain | card, analytics, validation, llm, obsidian, rag, generator, jobs | various |
| 5. Binaries | cli, api, mcp, worker | all crates |

## Key Rust Patterns

- Trait-based DI at every external boundary (DB, HTTP, Qdrant, Redis)
- `mockall` for auto-generated test mocks
- `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]` as default
- Newtype pattern for domain IDs (`struct NoteId(i64)`)
- `thiserror` enums per crate, unified via `From` conversions
- `#[instrument]` on async functions for tracing
- `Arc<AppState>` shared via axum State extractor

## Files Produced

- `specs/01-common.md` through `specs/19-worker.md` - one per crate
- `PROMPT.md` - ralph loop prompt
- `presets/tdd-rewrite.yml` - custom hat collection
- `ralph.yml` - ralph configuration
- `run-ralph.sh` - sequential execution script

# anki-atlas

Unified Anki flashcard platform: hybrid search index + card generation + obsidian sync + MCP tools. Written in Rust.

## Commands

- `cargo test --workspace` -- Run all tests. **Gate for every commit.**
- `cargo clippy --workspace -- -D warnings` -- Lint (must be clean)
- `cargo fmt --all -- --check` -- Format check
- `cargo build --release` -- Release build

Note: `database` and `anki-sync` crates require Docker for integration tests (testcontainers).
Skip them locally with `--exclude anki-sync --exclude database`.

## Architecture

```
crates/     -- Library crates (shared, reusable)
bins/       -- Binary entry points (cli, api, mcp, worker)
specs/      -- Ralph loop spec files (one per crate)
presets/    -- Ralph orchestrator configs (YAML + prompt files)
config/     -- Configuration files
```

**Dependency rule:** `bins/` depends on `crates/`, `crates/` never depends on `bins/`.
Crates may depend on other crates but **no circular dependencies**.

### Crates

| Crate | Purpose |
|-------|---------|
| common | Types, config (figment), errors (thiserror), tracing setup |
| taxonomy | Tag normalization and validation (500+ tag mappings) |
| database | PostgreSQL pool (sqlx), migrations |
| anki-reader | Anki SQLite reader (rusqlite), models, normalizer, AnkiConnect client |
| anki-sync | Sync engine, state tracking, progress, recovery |
| indexer | Embedding providers, Qdrant vector store, index service |
| search | Hybrid search: FTS + semantic + RRF fusion + reranking |
| analytics | Topic taxonomy, coverage, gap detection, duplicate detection |
| card | Card domain models, slug service, registry, APF format |
| validation | Validation pipeline with HTML/content/format/tag validators |
| llm | LLM provider abstraction (OpenRouter, Ollama) |
| obsidian | Vault parser, analyzer, frontmatter, sync workflow |
| rag | Document chunker, vector store, RAG service |
| generator | LLM-powered card generation agents, APF rendering |
| jobs | Background job queue (Redis via rustis) |

### Binaries

| Binary | Framework | Purpose |
|--------|-----------|---------|
| cli | clap | Command-line interface (12 subcommands) |
| api | axum | REST API with auth middleware |
| mcp | rmcp | MCP server for AI agents |
| worker | tokio | Background job worker |

### Ralph Presets

| Preset | Purpose | Run |
|--------|---------|-----|
| `tdd-rewrite.yml` | TDD loop for crate rewrites (red/green/refactor) | `ralph run -c presets/tdd-rewrite.yml` |
| `card-improve.yml` | Iterative card quality improvement (select/analyze/improve/review) | `ralph run -c presets/card-improve.yml` |
| `card-improve-openrouter.yml` | Card improvement via OpenRouter (opencode backend) | `ralph run -c presets/card-improve-openrouter.yml` |

## Git

- **Never** add `Co-Authored-By` trailers to commit messages
- Follow Conventional Commits (feat:, fix:, docs:, refactor:, chore:, test:)

## Conventions

- Rust 1.85+ (edition 2024)
- All types must be `Send + Sync`
- `thiserror` for library error types, `anyhow` only in binary crates
- Trait-based DI at every external boundary (DB, HTTP, Qdrant, Redis)
- `#[cfg_attr(test, mockall::automock)]` on traits for test mocks
- `#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]` as baseline
- Newtype pattern for domain IDs: `pub struct NoteId(pub i64);`
- `#[instrument]` on public async functions for tracing
- `Arc<T>` for shared state, never `Rc<T>`
- No `unwrap()` or `expect()` in library crates

## Testing

- Unit tests in `#[cfg(test)] mod tests` within each source file
- Integration tests in `crates/<name>/tests/` and `bins/<name>/tests/`
- `#[tokio::test]` for async tests
- `mockall` for auto-generated mock implementations
- `tempfile::TempDir` for filesystem tests
- Run single crate: `cargo test -p <crate-name>`

## Entry Points

- CLI: `cargo run --bin anki-atlas`
- API: `cargo run --bin anki-atlas-api`
- MCP: `cargo run --bin anki-atlas-mcp`
- Worker: `cargo run --bin anki-atlas-worker`

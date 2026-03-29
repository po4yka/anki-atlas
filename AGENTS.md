# anki-atlas

Unified Anki flashcard platform: hybrid search index + card generation + obsidian sync + MCP tools. Written in Rust.

## Commands

```bash
cargo test --workspace                    # Run all tests (gate for every commit)
cargo clippy --workspace -- -D warnings   # Lint (must be clean)
cargo fmt --all -- --check                # Format check
cargo build --release                     # Release build
```

Note: `database` and `anki-sync` crates require Docker (testcontainers). Skip locally with `--exclude anki-sync --exclude database`.

## Architecture

```
crates/     -- Library crates and shared runtime/support components
bins/       -- Product surfaces and tooling binaries
specs/      -- Ralph loop spec files (one per crate)
config/     -- Configuration files
```

**Dependency rule:** `bins/` depends on `crates/`, `crates/` never depends on `bins/`. No circular dependencies.

### Crates

| Crate | Purpose |
|-------|---------|
| common | Types, config (figment), errors (thiserror), tracing setup |
| taxonomy | Tag normalization and validation (500+ tag mappings) |
| database | PostgreSQL pool (sqlx), migrations |
| anki-reader | Anki SQLite reader (rusqlite), models, normalizer, AnkiConnect client |
| anki-sync | Sync engine, state tracking, progress, recovery |
| perf-support | Performance dataset seeding and support helpers |
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
| surface-contracts | Leaf-free DTOs shared by API, CLI, and MCP |
| surface-runtime | Shared runtime graph, facade composition, and local workflow wrappers |

### Runtime Surfaces

| Binary | Framework | Purpose |
|--------|-----------|---------|
| cli | clap + ratatui | Command-line surface plus local TUI operator console |
| api | axum | REST API with auth middleware |
| mcp | rmcp | MCP server for AI agents |
| worker | tokio | Background job worker |

### Tooling Binaries

| Binary | Framework | Purpose |
|--------|-----------|---------|
| perf-harness | goose | Load and performance harness |

## Conventions

- Rust 1.88+ (edition 2024)
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

## Anki Card Domain

This project manages Anki flashcards. Key domain concepts:

- **Bilingual cards**: Every concept gets EN + RU cards (Cyrillic only, no transliteration)
- **Slug format**: `{note_id}-{index}-{lang}` (e.g., `q-coroutines-0-en`)
- **Flat decks**: `Kotlin`, `Android`, `CompSci` -- no subdecks
- **Tags**: `prefix::topic` kebab-case, max 2 levels (see `docs/anki/tag-taxonomy.md`)
- **Quality**: Mastery-oriented (why/when/how, not "what is X"), atomic, under 100 words
- **CLI**: `cargo run --bin anki-atlas -- <command>`; see `README.md#cli-surface` for the current command inventory
- **MCP**: `ankiatlas_search`, `ankiatlas_validate`, `ankiatlas_sync`, etc.

Card-working skills: `.agents/skills/` (Codex) and `.claude/skills/` (Claude Code/OpenCode).
Reference docs: `docs/anki/`.

## Entry Points

```bash
cargo run --bin anki-atlas        # CLI
cargo run --bin anki-atlas-api    # API
cargo run --bin anki-atlas-mcp    # MCP
cargo run --bin anki-atlas-worker # Worker
```

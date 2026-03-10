# Anki Atlas

Unified Anki flashcard platform: hybrid search index + card generation + obsidian sync + MCP tools. Written in Rust.

## Features

- **Hybrid Search**: Combines semantic (vector) and keyword (FTS) search with RRF fusion
- **CrossEncoder Reranking**: Optional second-stage reranking of top hybrid candidates
- **Topic Coverage**: Analyze what topics your cards cover and identify gaps
- **Duplicate Detection**: Find near-duplicate cards using embedding similarity
- **Card Generation**: LLM-powered flashcard generation with APF format
- **Obsidian Sync**: Parse and sync notes from Obsidian vaults
- **Agent Tools**: MCP server for integration with AI agents (Claude Code, Claude Desktop)
- **Background Jobs**: Redis-backed async job queue for long-running operations

## Quick Start

### Prerequisites

- Rust 1.88+ (edition 2024)
- Docker and Docker Compose (for PostgreSQL, Qdrant, Redis)

### Setup

```bash
# Clone and enter directory
git clone https://github.com/po4yka/anki-atlas.git
cd anki-atlas

# Start dependencies (PostgreSQL + Qdrant + Redis)
docker compose -f infra/docker-compose.yml up -d

# Build the project
cargo build --release

# Run the CLI
cargo run --bin anki-atlas -- --help
cargo run --bin anki-atlas -- search "ownership" -n 5
cargo run --bin anki-atlas -- topics tree --root-path rust

# Run the API server
cargo run --bin anki-atlas-api

# Run the MCP server
cargo run --bin anki-atlas-mcp

# Run the background worker
ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1 cargo run --bin anki-atlas-worker
```

## Configuration

Set environment variables or use a config file:

| Variable | Description | Default |
|----------|-------------|---------|
| `ANKIATLAS_POSTGRES_URL` | PostgreSQL connection URL | `postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas` |
| `ANKIATLAS_QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `ANKIATLAS_REDIS_URL` | Redis URL for async jobs | `redis://localhost:6379/0` |
| `ANKIATLAS_EMBEDDING_PROVIDER` | `openai`, `google`, or `mock` | `openai` |
| `ANKIATLAS_EMBEDDING_MODEL` | Embedding model name | `text-embedding-3-small` |
| `ANKIATLAS_RERANK_ENABLED` | Enable CrossEncoder reranking | `false` |

## Development

```bash
# Run all tests (excludes Docker-dependent crates)
cargo test --workspace --exclude anki-sync --exclude database

# Lint
cargo clippy --workspace -- -D warnings

# Format
cargo fmt --all -- --check

# Build release
cargo build --release
```

## Project Structure

```
crates/          # Library crates (shared, reusable)
  common/        # Types, config, errors, tracing
  taxonomy/      # Tag normalization and validation
  database/      # PostgreSQL pool and migrations
  anki-reader/   # Anki SQLite reader and AnkiConnect client
  anki-sync/     # Sync engine with state tracking
  indexer/       # Embedding providers and Qdrant vector store
  search/        # Hybrid search with RRF fusion and reranking
  analytics/     # Topic coverage, gaps, duplicate detection
  card/          # Card domain models and registry
  validation/    # Validation pipeline
  llm/           # LLM provider abstraction (OpenRouter, Ollama)
  obsidian/      # Vault parser and sync workflow
  rag/           # Document chunker and RAG service
  generator/     # LLM-powered card generation agents
  jobs/          # Background job queue (Redis)
bins/            # Binary entry points
  cli/           # Command-line interface (clap)
  api/           # REST API (axum)
  mcp/           # MCP server (rmcp)
  worker/        # Background job worker (tokio)
```

## License

MIT

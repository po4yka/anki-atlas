# Anki Atlas

Anki Atlas is a Rust workspace for syncing Anki data, building a hybrid search index, analyzing topic coverage, previewing Obsidian-driven card generation, and exposing stable API, CLI, TUI, and MCP surfaces over the same `surface-runtime + surface-contracts` boundary.

## What Ships on `main`

| Surface | Binary | What it does |
|---|---|---|
| CLI | `anki-atlas` | Direct sync/index execution, search, analytics, taxonomy operations, and local preview workflows |
| TUI | `anki-atlas tui` | Full-screen operator console over the local runtime for search, analytics, and workflow execution |
| API | `anki-atlas-api` | Typed read endpoints plus async-only job entrypoints for sync and index |
| MCP | `anki-atlas-mcp` | Typed agent tools with `markdown` and `json` output modes |
| Worker | `anki-atlas-worker` | Redis-backed job execution, gated behind `ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1` |

Important current constraints:

- The HTTP API does not expose direct `/sync` or `/index` mutations.
- MCP does not run sync or index directly; it only enqueues jobs.
- `generate` is a preview workflow, not a persistence workflow.
- `obsidian-sync` is currently dry-run only; non-preview persistence fails explicitly.
- The root [Dockerfile](Dockerfile) builds `anki-atlas-api` only.

## Quick Start

### Prerequisites

- Rust `1.88+`
- Docker and Docker Compose
- PostgreSQL, Qdrant, and Redis
- Optional embedding credentials:
  - `OPENAI_API_KEY` for `ANKIATLAS_EMBEDDING_PROVIDER=openai`
  - `GEMINI_API_KEY` for `ANKIATLAS_EMBEDDING_PROVIDER=google` (`GOOGLE_API_KEY` still works)

### 1. Start infrastructure

```bash
docker compose -f infra/docker-compose.yml up -d
```

### 2. Configure the runtime

For a local smoke test without external model calls:

```bash
export ANKIATLAS_EMBEDDING_PROVIDER=mock
export ANKIATLAS_EMBEDDING_DIMENSION=384
```

For a real OpenAI-backed setup:

```bash
export ANKIATLAS_POSTGRES_URL=postgresql://ankiatlas:ankiatlas@localhost:5432/ankiatlas
export ANKIATLAS_QDRANT_URL=http://localhost:6333
export ANKIATLAS_REDIS_URL=redis://localhost:6379/0
export ANKIATLAS_EMBEDDING_PROVIDER=openai
export ANKIATLAS_EMBEDDING_MODEL=text-embedding-3-small
export ANKIATLAS_EMBEDDING_DIMENSION=1536
export OPENAI_API_KEY=sk-...
```

### 3. Build and migrate

```bash
cargo build
cargo run --bin anki-atlas -- migrate
```

### 4. Use the surfaces

```bash
# CLI
cargo run --bin anki-atlas -- search "ownership" -n 5
cargo run --bin anki-atlas -- topics tree --root-path rust
cargo run --bin anki-atlas -- tui

# API
cargo run --bin anki-atlas-api

# MCP
cargo run --bin anki-atlas-mcp

# Worker
ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1 cargo run --bin anki-atlas-worker
```

## API Surface

Current stable API routes:

- `GET /health`
- `GET /ready`
- `POST /jobs/sync`
- `POST /jobs/index`
- `GET /jobs/{job_id}`
- `POST /jobs/{job_id}/cancel`
- `POST /search`
- `POST /search/chunks`
- `GET /topics`
- `GET /topic-coverage`
- `GET /topic-gaps`
- `GET /topic-weak-notes`
- `GET /duplicates`

If `ANKIATLAS_API_KEY` is configured, all routes except `/health` and `/ready` require the `X-API-Key` header.

`/ready` currently reports process readiness only. It is not a deep dependency probe for PostgreSQL, Qdrant, or Redis.

`/search` remains the note-level hybrid endpoint. It now includes best semantic match metadata from the winning chunk on each note result. `/search/chunks` is a semantic-only raw chunk endpoint for multimodal hits.

## CLI Surface

The CLI currently supports:

- `version`
- `migrate`
- `tui`
- `sync`
- `index`
- `search`
- `topics tree`
- `topics load`
- `topics label`
- `coverage`
- `gaps`
- `weak-notes`
- `duplicates`
- `generate`
- `validate`
- `obsidian-sync`
- `tag-audit`

Example commands:

```bash
cargo run --bin anki-atlas -- sync /path/to/collection.anki2 --force-reindex
cargo run --bin anki-atlas -- duplicates --threshold 0.95 --max 25 --deck Rust
cargo run --bin anki-atlas -- validate /path/to/cards.txt --quality
cargo run --bin anki-atlas -- search "diagram" --chunks -n 10
cargo run --bin anki-atlas -- obsidian-sync /path/to/vault --dry-run
cargo run --bin anki-atlas -- tui
```

`search --chunks` returns raw chunk hits and stays semantic-only. `--fts` is not supported with `--chunks`.

### TUI

`anki-atlas tui` starts a full-screen terminal UI over the same direct local runtime used by the CLI. It bootstraps PostgreSQL, Qdrant, and Redis on startup and exposes:

- `Home` for bootstrap state, config summary, and session activity
- `Search` for hybrid search queries with deck/tag filters
- `Topics` for taxonomy tree, coverage, gaps, weak notes, and duplicates
- `Workflows` for sync, index, generate preview, validate, obsidian preview, and tag audit

Keybindings:

- `Tab` to advance within the active screen
- `Shift+Tab` to move focus back to the navigation rail
- `Arrow keys` or `j` / `k` to move
- `Arrow left/right` or `h` / `l` to change tabs or result selection
- `Enter` to edit, toggle, or run
- `/` to jump straight to search
- `Esc` to back out
- `q` to quit

## MCP Surface

The MCP server registers 15 tools:

- `ankiatlas_search`
- `ankiatlas_search_chunks`
- `ankiatlas_topics`
- `ankiatlas_topic_coverage`
- `ankiatlas_topic_gaps`
- `ankiatlas_topic_weak_notes`
- `ankiatlas_duplicates`
- `ankiatlas_sync_job`
- `ankiatlas_index_job`
- `ankiatlas_job_status`
- `ankiatlas_job_cancel`
- `ankiatlas_generate`
- `ankiatlas_validate`
- `ankiatlas_obsidian_sync`
- `ankiatlas_tag_audit`

Every tool accepts `output_mode = "markdown" | "json"`, defaulting to `markdown`.

`ankiatlas_search` stays note-oriented and includes best semantic match metadata on each result. `ankiatlas_search_chunks` returns raw multimodal chunk hits only.

## Configuration

Core environment variables come from [config.rs](crates/common/src/config.rs):

| Variable | Default | Notes |
|---|---|---|
| `ANKIATLAS_POSTGRES_URL` | `postgresql://localhost:5432/ankiatlas` | PostgreSQL connection |
| `ANKIATLAS_QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint |
| `ANKIATLAS_REDIS_URL` | `redis://localhost:6379/0` | Redis queue backend |
| `ANKIATLAS_JOB_QUEUE_NAME` | `ankiatlas_jobs` | Redis queue key |
| `ANKIATLAS_JOB_RESULT_TTL_SECONDS` | `86400` | Job metadata retention |
| `ANKIATLAS_JOB_MAX_RETRIES` | `3` | Job retry budget |
| `ANKIATLAS_EMBEDDING_PROVIDER` | `openai` | `openai`, `google`, or `mock` |
| `ANKIATLAS_EMBEDDING_MODEL` | `text-embedding-3-small` | Model name |
| `ANKIATLAS_EMBEDDING_DIMENSION` | `1536` | Must match provider expectations |
| `ANKIATLAS_RERANK_ENABLED` | `false` | Enables reranking if endpoint is also configured |
| `ANKIATLAS_RERANK_ENDPOINT` | none | Required when reranking is enabled |
| `ANKIATLAS_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker label |
| `ANKIATLAS_RERANK_TOP_N` | `50` | Candidate count for reranking |
| `ANKIATLAS_RERANK_BATCH_SIZE` | `32` | Reranker batch size |
| `ANKIATLAS_API_HOST` | `0.0.0.0` | API bind host |
| `ANKIATLAS_API_PORT` | `8000` | API bind port |
| `ANKIATLAS_API_KEY` | unset | Optional API auth |
| `ANKIATLAS_DEBUG` | `false` | Logging verbosity |
| `ANKIATLAS_ANKI_COLLECTION_PATH` | unset | Optional default collection path |
| `ANKIATLAS_ANKI_MEDIA_ROOT` | unset | Optional explicit Anki media root |

Gemini Embedding 2 notes:

- Use `ANKIATLAS_EMBEDDING_PROVIDER=google`.
- Use `ANKIATLAS_EMBEDDING_MODEL=gemini-embedding-2-preview`.
- `ANKIATLAS_EMBEDDING_DIMENSION` accepts any positive value up to `3072`; `3072`, `1536`, and `768` are the recommended sizes.
- Anki indexing stores one `text_primary` chunk plus supported local asset chunks for images, audio, video, and PDFs referenced from note fields.
- The media root resolves from `ANKIATLAS_ANKI_MEDIA_ROOT`, then sync metadata `last_collection_path`, then `ANKIATLAS_ANKI_COLLECTION_PATH`, with sibling `collection.media` as the derived default.
- Explicit `index` or `sync --force-reindex` work can recreate an incompatible Qdrant collection automatically. API and MCP startup stay read-only and return a `reindex required` error instead of mutating storage.

## Development

```bash
# format
cargo fmt --all -- --check

# lint
cargo clippy --workspace -- -D warnings

# focused tests
cargo test -p anki-atlas-api -p anki-atlas-cli -p anki-atlas-mcp

# broader workspace test run
cargo test --workspace --exclude anki-sync --exclude database
```

Fuzzing instructions live in [docs/FUZZING.md](docs/FUZZING.md).

## Documentation Map

- [Architecture](docs/ARCHITECTURE.md)
- [First Time Setup](docs/FIRST_TIME_SETUP.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Fuzzing](docs/FUZZING.md)
- [MCP Tools](docs/MCP_TOOLS.md)
- [Performance](docs/PERFORMANCE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [CLI Spec](specs/16-cli.md)
- [API Spec](specs/17-api.md)
- [MCP Spec](specs/18-mcp.md)

## Workspace Layout

```text
bins/
  api/       # axum API surface
  cli/       # clap CLI and ratatui TUI
  mcp/       # rmcp stdio server
  perf-harness/ # Goose-based performance runner
  worker/    # Redis-backed job worker
crates/
  analytics/
  anki-reader/
  anki-sync/
  card/
  common/
  database/
  generator/
  indexer/
  jobs/
  llm/
  obsidian/
  perf-support/
  rag/
  search/
  surface-contracts/
  surface-runtime/
  taxonomy/
  validation/
```

Product surfaces compose runtime behavior in `surface-runtime`, while the leaf-free DTO boundary shared by API, CLI, and MCP lives in `surface-contracts`. `perf-support` and `perf-harness` are support/tooling members, not user-facing runtime surfaces.

## License

MIT

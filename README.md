# Anki Atlas

Anki Atlas is a Rust workspace for syncing Anki data, building a hybrid search index, analyzing topic coverage, previewing Obsidian-driven card generation, and exposing stable API, CLI, and MCP surfaces over the same shared runtime.

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
- The root [Dockerfile](/Users/po4yka/GitRep/anki-atlas/Dockerfile) builds `anki-atlas-api` only.

## Quick Start

### Prerequisites

- Rust `1.88+`
- Docker and Docker Compose
- PostgreSQL, Qdrant, and Redis
- Optional embedding credentials:
  - `OPENAI_API_KEY` for `ANKIATLAS_EMBEDDING_PROVIDER=openai`
  - `GOOGLE_API_KEY` for `ANKIATLAS_EMBEDDING_PROVIDER=google`

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
- `GET /topics`
- `GET /topic-coverage`
- `GET /topic-gaps`
- `GET /topic-weak-notes`
- `GET /duplicates`

If `ANKIATLAS_API_KEY` is configured, all routes except `/health` and `/ready` require the `X-API-Key` header.

`/ready` currently reports process readiness only. It is not a deep dependency probe for PostgreSQL, Qdrant, or Redis.

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
cargo run --bin anki-atlas -- obsidian-sync /path/to/vault --dry-run
cargo run --bin anki-atlas -- tui
```

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

The MCP server registers 14 tools:

- `ankiatlas_search`
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

## Configuration

Core environment variables come from [config.rs](/Users/po4yka/GitRep/anki-atlas/crates/common/src/config.rs):

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

Fuzzing instructions live in [docs/FUZZING.md](/Users/po4yka/GitRep/anki-atlas/docs/FUZZING.md).

## Documentation Map

- [Architecture](/Users/po4yka/GitRep/anki-atlas/docs/ARCHITECTURE.md)
- [First Time Setup](/Users/po4yka/GitRep/anki-atlas/docs/FIRST_TIME_SETUP.md)
- [Deployment](/Users/po4yka/GitRep/anki-atlas/docs/DEPLOYMENT.md)
- [Fuzzing](/Users/po4yka/GitRep/anki-atlas/docs/FUZZING.md)
- [MCP Tools](/Users/po4yka/GitRep/anki-atlas/docs/MCP_TOOLS.md)
- [Performance](/Users/po4yka/GitRep/anki-atlas/docs/PERFORMANCE.md)
- [Troubleshooting](/Users/po4yka/GitRep/anki-atlas/docs/TROUBLESHOOTING.md)
- [CLI Spec](/Users/po4yka/GitRep/anki-atlas/specs/16-cli.md)
- [API Spec](/Users/po4yka/GitRep/anki-atlas/specs/17-api.md)
- [MCP Spec](/Users/po4yka/GitRep/anki-atlas/specs/18-mcp.md)

## Workspace Layout

```text
bins/
  api/       # axum API surface
  cli/       # clap-based CLI
  mcp/       # rmcp stdio server
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
  rag/
  search/
  surface-runtime/
  taxonomy/
  validation/
```

## License

MIT

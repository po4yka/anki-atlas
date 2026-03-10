# Anki Atlas Architecture

## Mission

Anki Atlas is a Rust workspace for syncing Anki collections, indexing note content for hybrid retrieval, analyzing topic coverage, previewing Obsidian-driven workflows, and exposing those capabilities consistently through API, CLI, MCP, and worker processes.

The core architecture rule on `main` is simple:

- public interfaces must match wired domain services
- unsupported workflows must fail explicitly
- shared runtime composition belongs in `crates/surface-runtime`

## Current Stack

| Layer | Technology |
|---|---|
| Language | Rust 2024 |
| API | Axum |
| CLI | Clap |
| MCP | rmcp |
| Jobs | Tokio + Redis |
| Relational storage | PostgreSQL |
| Vector storage | Qdrant |

## Workspace Layout

```text
bins/
  api/       # HTTP surface
  cli/       # command-line surface
  mcp/       # stdio MCP server
  worker/    # async job execution process
crates/
  common/           # config, logging, shared errors and types
  database/         # PostgreSQL pool and migrations
  anki-reader/      # Anki SQLite + AnkiConnect access
  anki-sync/        # sync orchestration
  indexer/          # embeddings and vector persistence
  search/           # hybrid search and reranking
  analytics/        # taxonomy, coverage, gaps, duplicates
  obsidian/         # note parsing and vault analysis
  validation/       # validation pipeline and quality scoring
  generator/        # generation models and agents
  jobs/             # job types, persistence, queue manager
  surface-runtime/  # shared runtime graph and local workflow wrappers
```

Older Python-oriented specs and migration notes still exist under `specs/` and `docs/plans/`, but they are historical design artifacts unless they have been explicitly rewritten for the current Rust runtime.

## Layer Boundaries

```text
Surfaces
  bins/api
  bins/cli
  bins/mcp
  bins/worker
        |
        v
Shared runtime composition
  crates/surface-runtime
        |
        v
Domain services
  anki-sync
  indexer
  search
  analytics
  obsidian
  validation
  generator
  jobs
        |
        v
Infrastructure
  PostgreSQL
  Qdrant
  Redis
  embedding providers
```

### Ownership rules

- `bins/*` should extract input, call facades, and translate output.
- `crates/*` should contain business logic and reusable workflows.
- Shared runtime wiring should not be duplicated across API, CLI, and MCP.
- Write-side transport differences are allowed only when intentional:
  - CLI may execute sync/index directly.
  - API and MCP must keep sync/index behind jobs.

## Shared Runtime

[services.rs](/Users/po4yka/GitRep/anki-atlas/crates/surface-runtime/src/services.rs) builds the runtime graph once from [config.rs](/Users/po4yka/GitRep/anki-atlas/crates/common/src/config.rs):

- PostgreSQL pool
- embedding provider
- Qdrant-backed vector repository
- optional reranker
- Redis-backed `JobManager`
- `SearchFacade`
- `AnalyticsFacade`
- direct sync/index executors for CLI-only use
- local workflow wrappers for generation preview, validation, Obsidian scan, and tag audit

This lets the surfaces share the same domain contracts while keeping their transport code thin.

## Stable Public Surfaces

### HTTP API

The API exposes:

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

Important behavior:

- Direct `/sync` and `/index` mutations are intentionally absent.
- If `ANKIATLAS_API_KEY` is set, all routes except `/health` and `/ready` require `X-API-Key`.
- `X-Request-ID` is added to every response.
- `/ready` currently signals process readiness only. It is not a deep dependency check.
- Error responses use one JSON envelope:

```json
{
  "error": "BadRequest",
  "message": "limit must be greater than 0",
  "details": {}
}
```

### CLI

The CLI exposes:

- `version`
- `migrate`
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

Important behavior:

- `sync` and `index` execute directly through the shared runtime.
- `generate` is preview-only.
- `obsidian-sync` requires `--dry-run` today because persistence is not implemented.

### MCP

The MCP server exposes:

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

Important behavior:

- Every tool supports `output_mode = "markdown" | "json"`.
- Markdown is the default.
- JSON returns the same canonical structured payload as markdown mode.
- Sync and index are async job tools only.

### Worker

`anki-atlas-worker` consumes the Redis queue and executes job bodies. It is currently gated by `ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1` while worker execution continues to harden.

## Primary Flows

### Async sync and index flow

```text
client
  -> API or MCP job tool
  -> jobs::JobManager
  -> Redis queue
  -> worker
  -> jobs::tasks
  -> anki-sync / indexer / storage backends
```

### Direct CLI execution flow

```text
CLI
  -> surface-runtime direct executor
  -> anki-sync / indexer
  -> PostgreSQL + Qdrant
```

### Local preview flow

```text
CLI or MCP
  -> surface-runtime workflow wrapper
  -> obsidian / validation / taxonomy crates
```

## Operational Constraints

These are intentional today, not accidental omissions:

- no direct synchronous HTTP `/sync` or `/index`
- no direct sync/index execution from MCP
- no Obsidian persistence path outside preview mode
- no worker execution unless explicitly enabled

If any of those constraints change, the code, tests, and docs should change together.

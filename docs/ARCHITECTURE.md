# Anki Atlas Architecture

## Mission

Anki Atlas is a Rust workspace for syncing Anki collections, indexing note content and supported Anki media for hybrid and chunk retrieval, analyzing topic coverage, previewing Obsidian-driven workflows, and exposing those capabilities consistently through API, CLI, MCP, and worker processes.

The core architecture rule on `main` is simple:

- public interfaces must match wired domain services
- unsupported workflows must fail explicitly
- shared surface composition belongs in `crates/surface-runtime` and `crates/surface-contracts`

## Current Stack

| Layer | Technology |
|---|---|
| Language | Rust 2024 |
| API | Axum |
| CLI / TUI | Clap + Ratatui |
| MCP | rmcp |
| Jobs | Tokio + Redis |
| Relational storage | PostgreSQL |
| Vector storage | Qdrant |

## Workspace Layout

```text
bins/
  api/       # HTTP surface
  cli/       # command-line and TUI surface
  mcp/       # stdio MCP server
  perf-harness/ # Goose performance runner
  worker/    # async job execution process
crates/
  analytics/        # taxonomy, coverage, gaps, duplicates
  anki-reader/      # Anki SQLite + AnkiConnect access
  anki-sync/        # sync orchestration
  card/             # card domain models, registry, APF
  common/           # config, logging, shared errors and types
  database/         # PostgreSQL pool and migrations
  generator/        # generation models and agents
  indexer/          # embeddings and vector persistence
  jobs/             # job types, persistence, queue manager
  llm/              # LLM provider abstraction
  obsidian/         # note parsing and vault analysis
  perf-support/     # deterministic perf datasets and helpers
  rag/              # chunking and retrieval support
  search/           # hybrid search and reranking
  surface-contracts/ # shared DTOs for API, CLI, MCP
  surface-runtime/  # shared runtime graph and local workflow wrappers
  taxonomy/         # tag normalization and validation
  validation/       # validation pipeline and quality scoring
```

Older Python-oriented specs and migration notes still exist under `specs/` and `docs/plans/`, but they are historical design artifacts unless they have been explicitly rewritten for the current Rust runtime.

## Layer Boundaries

```text
Public runtime surfaces
  bins/api
  bins/cli (subcommands + tui)
  bins/mcp
  bins/worker
        |
        v
Shared surface boundary
  crates/surface-contracts
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

Support and tooling
  crates/perf-support
  bins/perf-harness
```

### Ownership rules

- `bins/*` should extract input, call facades, and translate output.
- `crates/*` should contain business logic and reusable workflows.
- Shared runtime wiring should not be duplicated across API, CLI, and MCP.
- `surface-contracts` owns the leaf-free surface DTOs used by API, CLI, and MCP.
- `surface-runtime` owns contract-to-domain mapping and runtime composition.
- Write-side transport differences are allowed only when intentional:
  - CLI may execute sync/index directly.
  - API and MCP must keep sync/index behind jobs.

## Shared Runtime

[services.rs](crates/surface-runtime/src/services.rs) builds the runtime graph once from [config.rs](crates/common/src/config.rs), and [surface-contracts](crates/surface-contracts/src/lib.rs) provides the shared leaf-free DTOs consumed by API, CLI, and MCP:

- PostgreSQL pool
- embedding provider
- Qdrant-backed vector repository
- read-only vector compatibility validation for API and MCP bootstrap
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
- `POST /search/chunks`
- `GET /topics`
- `GET /topic-coverage`
- `GET /topic-gaps`
- `GET /topic-weak-notes`
- `GET /duplicates`

Important behavior:

- Direct `/sync` and `/index` mutations are intentionally absent.
- `/search` remains note-level hybrid retrieval and attaches best semantic chunk metadata when semantic matches are present.
- `/search/chunks` is semantic-only and returns raw chunk hits for multimodal search.
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

### CLI and TUI

The CLI exposes:

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

Important behavior:

- `sync` and `index` execute directly through the shared runtime.
- `tui` runs over the same direct local runtime and focuses on search, topics, and workflow execution.
- explicit `index` and sync+index runs may recreate an incompatible vector collection when the embedding model, dimension, or vector schema has changed
- `search --chunks` exposes semantic-only raw chunk search; standard `search` remains note-level hybrid search
- `generate` is preview-only.
- `obsidian-sync` requires `--dry-run` today because persistence is not implemented.

### MCP

The MCP server exposes:

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

Important behavior:

- Every tool supports `output_mode = "markdown" | "json"`.
- Markdown is the default.
- JSON returns the same canonical structured payload as markdown mode.
- Sync and index are async job tools only.
- `ankiatlas_search_chunks` is semantic-only and returns raw multimodal chunk hits.

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

### Multimodal indexing model

```text
Anki note
  -> normalized_text
  -> text_primary chunk
  -> media refs from fields_json/raw_fields
  -> asset chunks (image/audio/video/document)
  -> Qdrant payloads keyed by stable chunk IDs
```

Important behavior:

- text and asset chunks share a note-level content hash derived from the embedding model and all chunk hash parts
- media root resolution uses `ANKIATLAS_ANKI_MEDIA_ROOT`, then sync metadata `last_collection_path`, then the configured collection path, deriving sibling `collection.media`
- duplicate detection and note-to-note similarity stay filtered to `text_primary` chunks

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
- API and MCP bootstrap do not auto-recreate incompatible vector collections; they fail with `reindex required` until an explicit index path runs

If any of those constraints change, the code, tests, and docs should change together.

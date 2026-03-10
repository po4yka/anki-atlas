# Anki Atlas Architecture

## Mission

Anki Atlas is a Rust workspace for ingesting Anki and Obsidian content, generating cards, building searchable indexes, and exposing a small set of stable machine-facing surfaces.

The important architecture rule on `main` is simple: public interfaces must match wired domain services. The API, CLI, and MCP surfaces now share one runtime composition layer in `crates/surface-runtime`, so transport-specific code stays thin while the domain wiring stays consistent.

## Current Stack

| Layer | Technology |
|-------|------------|
| Language | Rust 2024 |
| API | Axum |
| CLI | Clap |
| MCP | rmcp |
| Jobs | Tokio + Redis |
| Relational store | PostgreSQL |
| Vector store | Qdrant |

## Workspace Layout

```text
bins/
  api/       # HTTP surface
  cli/       # command-line surface
  mcp/       # MCP server
  worker/    # background job executor
crates/
  common/      # config, errors, logging, shared types
  anki-reader/ # Anki SQLite + AnkiConnect access
  anki-sync/   # sync orchestration
  indexer/     # embeddings + Qdrant persistence
  search/      # hybrid retrieval and reranking
  analytics/   # taxonomy, coverage, gaps, duplicates
  obsidian/    # vault discovery and parsing
  generator/   # card generation agents
  llm/         # LLM provider contracts
  jobs/        # job queue contracts, persistence, manager
  surface-runtime/ # shared runtime graph and local workflow wrappers
```

Legacy `apps/` and `packages/` directories are historical leftovers from the earlier rewrite and are not the authoritative runtime architecture.

## Layer Boundaries

```text
Interfaces
  bins/api
  bins/cli
  bins/mcp
  bins/worker
        |
        v
Application and domain crates
  anki-sync
  indexer
  search
  analytics
  obsidian
  generator
  jobs
        |
        v
Infrastructure and shared contracts
  common
  anki-reader
  llm
  PostgreSQL
  Qdrant
  Redis
```

### Ownership rules

- `bins/*` should translate transport concerns only.
- Business rules belong in `crates/*`.
- Shared domain behavior should be reused across surfaces, not reimplemented in each binary.
- Unsupported workflows should fail explicitly or stay unexposed.

## Stable Public Surfaces

### HTTP API

The API exposes health checks, async job orchestration, and service-aligned read operations:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | `GET` | liveness |
| `/ready` | `GET` | dependency readiness |
| `/jobs/sync` | `POST` | enqueue sync job |
| `/jobs/index` | `POST` | enqueue index job |
| `/jobs/{job_id}` | `GET` | poll job status |
| `/jobs/{job_id}/cancel` | `POST` | request cancellation |
| `/search` | `POST` | hybrid retrieval via `search::service::SearchService` |
| `/topics` | `GET` | taxonomy / coverage tree |
| `/topic-coverage` | `GET` | topic coverage metrics |
| `/topic-gaps` | `GET` | undercovered and missing topic list |
| `/topic-weak-notes` | `GET` | weak notes within a topic subtree |
| `/duplicates` | `GET` | near-duplicate note clusters |

The API does not expose direct `/sync` or `/index` mutations. Write-side ingestion stays behind `/jobs/*`.

#### Route semantics

- `/search` accepts a typed JSON body that mirrors `search::service::SearchParams`.
- `/topics` accepts optional `root_path`.
- `/topic-coverage` requires `topic_path` and accepts optional `include_subtree`.
- `/topic-gaps` requires `topic_path` and accepts optional `min_coverage`.
- `/topic-weak-notes` requires `topic_path` and accepts optional `max_results`.
- `/duplicates` accepts optional `threshold`, `max_clusters`, `deck_filter[]`, and `tag_filter[]`.

### CLI

The CLI exposes the real service-aligned command surface:

| Command | Purpose |
|---------|---------|
| `version` | print build version |
| `migrate` | run database migrations |
| `sync` | sync an Anki collection directly and optionally reindex |
| `index` | run direct indexing against PostgreSQL notes |
| `search` | run hybrid retrieval from the terminal |
| `topics tree` | print the taxonomy / coverage tree |
| `topics load` | load taxonomy YAML into PostgreSQL |
| `topics label` | label notes against the taxonomy |
| `coverage` | inspect topic coverage |
| `gaps` | inspect undercovered and missing topics |
| `weak-notes` | list weak notes for a topic |
| `duplicates` | inspect duplicate-note clusters |
| `generate` | parse an Obsidian note and preview card generation |
| `validate` | validate flashcard content from a file |
| `obsidian-sync` | scan an Obsidian vault in preview mode |
| `tag-audit` | audit and optionally normalize tag conventions |

### Worker

`anki-atlas-worker` is the only process allowed to execute background job bodies. It is intentionally gated behind `ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1` until task execution is fully implemented end to end.

### MCP

The MCP server exposes the same read and local-preview capabilities through typed tools. Write-side work remains async-only through job tools.

Registered tools:

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

Each tool accepts `output_mode = "markdown" | "json"`. Markdown is the default, while JSON returns the same structured payload for programmatic use.

## Primary Flows

### Async sync and indexing

```text
client
  -> bins/api
  -> crates/jobs::JobManager
  -> Redis queue
  -> bins/worker
  -> crates/jobs::tasks
  -> domain crates and storage backends
```

### Obsidian generation flow

```text
CLI or MCP
  -> crates/surface-runtime workflow wrappers
  -> obsidian / validation / taxonomy / generator crates
```

## Design Constraints

- Keep interface contracts honest. Do not ship placeholder success responses.
- Prefer one translation layer per surface, mapped from shared domain types.
- Avoid parallel public entrypoints for the same domain behavior unless they share one underlying service.
- Treat docs as part of the architecture contract. They must describe the compiled runtime, not the intended future state.

## Near-Term Gaps

These capabilities still remain intentionally constrained on `main`:

- direct synchronous `/sync` and `/index` HTTP mutations
- direct sync/index execution from MCP tools
- fully stable background worker execution

Those flows should be expanded only when the transport surface and the underlying shared services stay aligned.

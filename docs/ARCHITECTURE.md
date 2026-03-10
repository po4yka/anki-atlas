# Anki Atlas Architecture

## Mission

Anki Atlas is a Rust workspace for ingesting Anki and Obsidian content, generating cards, building searchable indexes, and exposing a small set of stable machine-facing surfaces.

The important architecture rule on `main` is simple: public interfaces must match wired domain services. If a flow is not backed by a shared service with tests, it should stay out of the exposed API and CLI.

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

The API currently exposes only health checks and async job orchestration:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | `GET` | liveness |
| `/ready` | `GET` | dependency readiness |
| `/jobs/sync` | `POST` | enqueue sync job |
| `/jobs/index` | `POST` | enqueue index job |
| `/jobs/{job_id}` | `GET` | poll job status |
| `/jobs/{job_id}/cancel` | `POST` | request cancellation |

Direct `/sync`, `/index`, `/search`, `/topics`, `/duplicates`, and `/index/info` routes are intentionally not part of the `main` contract because they are not yet backed by one shared service layer.

### CLI

The CLI currently exposes only commands with real local implementations:

| Command | Purpose |
|---------|---------|
| `version` | print build version |
| `migrate` | run database migrations |
| `generate` | parse an Obsidian note and preview card generation |
| `validate` | validate flashcard content from a file |
| `obsidian-sync` | scan an Obsidian vault and preview or sync cards |
| `tag-audit` | audit tag conventions |

Search, sync, topics, coverage, gaps, duplicates, and direct indexing commands are not exported on `main`.

### Worker

`anki-atlas-worker` is the only process allowed to execute background job bodies. It is intentionally gated behind `ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1` until task execution is fully implemented end to end.

### MCP

The MCP package is present in the workspace, but its public tool contract must stay narrower than the internal crate graph. Only tools with real handler implementations should be advertised.

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
  -> obsidian parsing crates
  -> generator agents
  -> validation / formatting
```

## Design Constraints

- Keep interface contracts honest. Do not ship placeholder success responses.
- Prefer one translation layer per surface, mapped from shared domain types.
- Avoid parallel public entrypoints for the same domain behavior unless they share one underlying service.
- Treat docs as part of the architecture contract. They must describe the compiled runtime, not the intended future state.

## Near-Term Gaps

These capabilities exist in crates but are not yet promoted to stable public surfaces:

- direct hybrid search HTTP and CLI
- taxonomy coverage and gap endpoints
- duplicate detection endpoints
- synchronous sync and index entrypoints
- fully stable background worker execution

Those flows should be re-exposed only after wiring through shared service APIs and adding transport-level tests.

# Spec: crate `anki-atlas-api`

## Source Reference

Current Rust source:

- [main.rs](/Users/po4yka/GitRep/anki-atlas/bins/api/src/main.rs)
- [router.rs](/Users/po4yka/GitRep/anki-atlas/bins/api/src/router.rs)
- [schemas.rs](/Users/po4yka/GitRep/anki-atlas/bins/api/src/schemas.rs)
- [error.rs](/Users/po4yka/GitRep/anki-atlas/bins/api/src/error.rs)
- [handlers/](/Users/po4yka/GitRep/anki-atlas/bins/api/src/handlers)
- [state.rs](/Users/po4yka/GitRep/anki-atlas/bins/api/src/state.rs)

## Purpose

Expose a service-aligned HTTP v2 surface for Anki Atlas:

- health endpoints
- async-only sync/index job orchestration
- typed synchronous read endpoints for search and analytics

Direct synchronous `/sync` and `/index` HTTP mutations are intentionally absent.

## Dependencies

```toml
[dependencies]
common = { path = "../../crates/common" }
jobs = { path = "../../crates/jobs" }
surface-contracts = { path = "../../crates/surface-contracts" }
surface-runtime = { path = "../../crates/surface-runtime" }

anyhow.workspace = true
async-trait.workspace = true
axum.workspace = true
chrono.workspace = true
serde.workspace = true
serde_json.workspace = true
sqlx.workspace = true
tokio.workspace = true
tower.workspace = true
tower-http.workspace = true
tracing.workspace = true
uuid.workspace = true
```

## Public API

### Routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness and version |
| `GET` | `/ready` | local process readiness |
| `POST` | `/jobs/sync` | enqueue sync job |
| `POST` | `/jobs/index` | enqueue index job |
| `GET` | `/jobs/{job_id}` | inspect job |
| `POST` | `/jobs/{job_id}/cancel` | request cancellation |
| `POST` | `/search` | typed hybrid search |
| `POST` | `/search/chunks` | semantic-only raw chunk search |
| `GET` | `/topics` | taxonomy tree |
| `GET` | `/topic-coverage` | topic coverage |
| `GET` | `/topic-gaps` | topic gaps |
| `GET` | `/topic-weak-notes` | weak-note list |
| `GET` | `/duplicates` | duplicate clusters |

### Auth and Middleware

- `/health` and `/ready` are always public
- all other routes require `X-API-Key` only when `ANKIATLAS_API_KEY` is configured
- every response includes `X-Request-ID`

### Application State

`AppState` carries:

- extracted API settings
- shared `SurfaceServices`

The API builds those services through [surface-runtime](/Users/po4yka/GitRep/anki-atlas/crates/surface-runtime/src/services.rs) with direct execution disabled.
The shared request and response DTOs for search and analytics come from [surface-contracts](/Users/po4yka/GitRep/anki-atlas/crates/surface-contracts/src/lib.rs).

### Request DTOs

Current request DTO families:

- `AsyncSyncRequest`
- `AsyncIndexRequest`
- `SearchRequest` from `surface-contracts`
- `ChunkSearchRequest` from `surface-contracts`
- query DTOs for topics, coverage, gaps, weak notes, and duplicates

Important `SearchRequest` fields:

- `query`
- `filters`
- `limit`
- `semantic_weight`
- `fts_weight`
- `semantic_only`
- `fts_only`
- `rerank_override`
- `rerank_top_n_override`

Important `ChunkSearchRequest` fields:

- `query`
- `filters`
- `limit`

### Response DTOs

Current response DTO families:

- `HealthResponse`
- `ReadyResponse`
- `JobAcceptedResponse`
- `JobStatusResponse`
- `SearchResponse` from `surface-contracts`
- `ChunkSearchResponse` from `surface-contracts`
- `TopicsTreeResponse`
- `TopicCoverageResponse`
- `TopicGapsResponse`
- `TopicWeakNotesResponse`
- `DuplicatesResponse`

### Error Contract

All handler and domain errors are translated to a common JSON shape:

```json
{
  "error": "NotFound",
  "message": "topic not found: rust/ownership",
  "details": {}
}
```

Status mapping:

- `400` for request validation and bad input
- `404` for missing domain entities such as topic paths
- `409` for explicit unsupported or conflicting job states
- `503` for database, vector store, or job backend unavailability
- `500` for unclassified internal failures

## Runtime Rules

- API read endpoints call shared search and analytics facades
- `/search` stays note-level hybrid search and attaches best semantic match metadata (`match_modality`, `match_chunk_kind`, `match_source_field`, `match_asset_rel_path`, `match_preview_label`) when semantic retrieval contributes
- `/search/chunks` is semantic-only and returns raw chunk hits with `chunk_id`, `chunk_kind`, `modality`, `source_field`, `asset_rel_path`, `mime_type`, `preview_label`, and `score`
- API write-side work is job-based only
- handlers should translate DTOs only and not construct service dependencies
- `/ready` does not perform deep dependency checks
- API bootstrap is read-only with respect to vector storage; when the collection model, dimension, or vector schema is incompatible, startup fails with a clear `reindex required` error

## Module Layout

```text
bins/api/src/
  main.rs
  lib.rs
  router.rs
  state.rs
  services.rs
  schemas.rs
  error.rs
  middleware.rs
  handlers/
    duplicates.rs
    health.rs
    jobs.rs
    search.rs
    topics.rs
```

## Acceptance Criteria

- router exposes only the documented v2 routes
- no direct `/sync`, `/index`, or `/index/info` routes exist
- all public DTOs map to real service contracts
- auth and correlation-id middleware remain intact
- docs do not describe unwired placeholder handlers

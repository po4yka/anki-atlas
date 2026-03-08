# Spec: crate `api`

## Source Reference
Python: `apps/api/main.py` + `apps/api/schemas.py`

## Purpose
HTTP REST API server exposing all anki-atlas operations: sync, index, search, topic analytics, duplicate detection, and background job management. Built on axum with tower middleware for correlation IDs, optional API key authentication, and structured error responses. Provides health/readiness probes, synchronous and async (job-based) endpoints.

## Dependencies
```toml
[dependencies]
common = { path = "../common" }
jobs = { path = "../jobs" }
# Additional workspace crates: anki-sync, indexer, search, analytics

anyhow = "1"
axum = { version = "0.8", features = ["json", "query"] }
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "signal"] }
tower = { version = "0.5", features = ["timeout"] }
tower-http = { version = "0.6", features = ["cors", "trace", "request-id"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
uuid = { version = "1", features = ["v4"] }

[dev-dependencies]
axum-test = "16"
serde_json = "1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Public API

### Application State (`src/state.rs`)

```rust
use std::sync::Arc;
use common::config::Settings;

/// Shared application state accessible in all handlers.
#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<Settings>,
    pub job_manager: Arc<dyn jobs::JobManager>,
    // Additional service handles (search, analytics, etc.)
}
```

### Request/Response Schemas (`src/schemas.rs`)

```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// --- Sync ---

#[derive(Debug, Deserialize)]
pub struct SyncRequest {
    pub source: String,
    #[serde(default = "default_true")]
    pub run_migrations: bool,
    #[serde(default = "default_true")]
    pub index: bool,
    #[serde(default)]
    pub force_reindex: bool,
}

#[derive(Debug, Serialize)]
pub struct SyncResponse {
    pub status: String,
    pub decks_upserted: i64,
    pub models_upserted: i64,
    pub notes_upserted: i64,
    pub notes_deleted: i64,
    pub cards_upserted: i64,
    pub card_stats_upserted: i64,
    pub duration_ms: i64,
    pub notes_embedded: Option<i64>,
    pub notes_skipped: Option<i64>,
    pub index_errors: Option<Vec<String>>,
}

// --- Index ---

#[derive(Debug, Deserialize)]
pub struct IndexRequest {
    #[serde(default)]
    pub force_reindex: bool,
}

#[derive(Debug, Serialize)]
pub struct IndexResponse {
    pub status: String,
    pub notes_processed: i64,
    pub notes_embedded: i64,
    pub notes_skipped: i64,
    pub notes_deleted: i64,
    pub errors: Vec<String>,
}

// --- Async Jobs ---

#[derive(Debug, Deserialize)]
pub struct AsyncSyncRequest {
    pub source: String,
    #[serde(default = "default_true")]
    pub run_migrations: bool,
    #[serde(default = "default_true")]
    pub index: bool,
    #[serde(default)]
    pub force_reindex: bool,
    pub run_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize)]
pub struct AsyncIndexRequest {
    #[serde(default)]
    pub force_reindex: bool,
    pub run_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize)]
pub struct JobAcceptedResponse {
    pub job_id: String,
    pub status: String,
    pub job_type: String,
    pub created_at: DateTime<Utc>,
    pub scheduled_for: Option<DateTime<Utc>>,
    pub poll_url: String,
}

#[derive(Debug, Serialize)]
pub struct JobStatusResponse {
    pub job_id: String,
    pub job_type: String,
    pub status: String,
    pub progress: f64,
    pub message: Option<String>,
    pub attempts: u32,
    pub max_retries: u32,
    pub cancel_requested: bool,
    pub created_at: Option<DateTime<Utc>>,
    pub scheduled_for: Option<DateTime<Utc>>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub result: Option<HashMap<String, serde_json::Value>>,
    pub error: Option<String>,
}

// --- Search ---

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub deck_names: Option<Vec<String>>,
    pub deck_names_exclude: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub tags_exclude: Option<Vec<String>>,
    pub model_ids: Option<Vec<i64>>,
    pub min_ivl: Option<i32>,
    pub max_lapses: Option<i32>,
    pub min_reps: Option<i32>,
    #[serde(default = "default_20")]
    pub limit: usize,
    #[serde(default = "default_one")]
    pub semantic_weight: f64,
    #[serde(default = "default_one")]
    pub fts_weight: f64,
}

#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    pub note_id: i64,
    pub rrf_score: f64,
    pub semantic_score: Option<f64>,
    pub semantic_rank: Option<i32>,
    pub fts_score: Option<f64>,
    pub fts_rank: Option<i32>,
    pub headline: Option<String>,
    pub rerank_score: Option<f64>,
    pub rerank_rank: Option<i32>,
    pub sources: Vec<String>,
    pub normalized_text: Option<String>,
    pub tags: Option<Vec<String>>,
    pub deck_names: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub results: Vec<SearchResultItem>,
    pub stats: HashMap<String, i64>,
    pub filters_applied: HashMap<String, serde_json::Value>,
    pub lexical: Option<HashMap<String, serde_json::Value>>,
    pub rerank: Option<HashMap<String, serde_json::Value>>,
}

// --- Topics/Coverage/Gaps ---

#[derive(Debug, Serialize)]
pub struct TopicCoverageResponse {
    pub topic_id: i64,
    pub path: String,
    pub label: String,
    pub note_count: i64,
    pub subtree_count: i64,
    pub child_count: i64,
    pub covered_children: i64,
    pub mature_count: i64,
    pub avg_confidence: f64,
    pub weak_notes: i64,
    pub avg_lapses: f64,
}

#[derive(Debug, Serialize)]
pub struct TopicGapItem {
    pub topic_id: i64,
    pub path: String,
    pub label: String,
    pub description: Option<String>,
    pub gap_type: String,
    pub note_count: i64,
    pub threshold: i64,
}

#[derive(Debug, Serialize)]
pub struct TopicGapsResponse {
    pub root_path: String,
    pub min_coverage: i64,
    pub gaps: Vec<TopicGapItem>,
    pub missing_count: i64,
    pub undercovered_count: i64,
}

// --- Duplicates ---

#[derive(Debug, Serialize)]
pub struct DuplicateNoteItem {
    pub note_id: i64,
    pub similarity: f64,
    pub text: String,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DuplicateClusterItem {
    pub representative_id: i64,
    pub representative_text: String,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
    pub duplicates: Vec<DuplicateNoteItem>,
    pub size: usize,
}

#[derive(Debug, Serialize)]
pub struct DuplicatesResponse {
    pub clusters: Vec<DuplicateClusterItem>,
    pub stats: HashMap<String, serde_json::Value>,
}

fn default_true() -> bool { true }
fn default_20() -> usize { 20 }
fn default_one() -> f64 { 1.0 }
```

### Router (`src/router.rs`)

```rust
use axum::{Router, routing::{get, post}};
use crate::state::AppState;

/// Build the application router with all routes.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // Health
        .route("/health", get(handlers::health))
        .route("/ready", get(handlers::ready))
        // Sync operations
        .route("/sync", post(handlers::sync))
        .route("/index", post(handlers::index_notes))
        // Async jobs
        .route("/jobs/sync", post(handlers::enqueue_sync_job))
        .route("/jobs/index", post(handlers::enqueue_index_job))
        .route("/jobs/{job_id}", get(handlers::get_job_status))
        .route("/jobs/{job_id}/cancel", post(handlers::cancel_job))
        // Search
        .route("/search", post(handlers::search))
        // Topics
        .route("/topics", get(handlers::list_topics))
        .route("/topics/{*topic_path}/coverage", get(handlers::topic_coverage))
        .route("/topics/{*topic_path}/gaps", get(handlers::topic_gaps))
        // Duplicates
        .route("/duplicates", get(handlers::find_duplicates))
        // Index info
        .route("/index/info", get(handlers::index_info))
        .with_state(state)
}
```

### Middleware (`src/middleware.rs`)

```rust
use axum::http::{Request, Response, HeaderValue};
use tower::Layer;

/// Correlation ID middleware: reads X-Request-ID header (or generates UUID),
/// sets it in tracing span, and includes it in response header.
pub struct CorrelationIdLayer;

/// Optional API key authentication middleware.
/// Reads X-API-Key header, compares against configured key.
/// Skips auth if no key is configured.
pub struct ApiKeyLayer {
    api_key: Option<String>,
}
```

### Error Handling (`src/error.rs`)

```rust
use axum::response::{IntoResponse, Response};
use axum::http::StatusCode;
use serde_json::json;

/// Map domain errors to HTTP status codes:
/// - NotFoundError -> 404
/// - ConflictError -> 409
/// - DatabaseError, VectorStoreError -> 503
/// - JobBackendUnavailableError -> 503
/// - Other AnkiAtlasError -> 500
pub struct AppError(pub anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response;
}
```

### Module structure

```
src/
  main.rs          -- Server startup, signal handling
  state.rs         -- AppState
  router.rs        -- Route definitions
  schemas.rs       -- Request/response types
  middleware.rs     -- CorrelationId, ApiKey layers
  error.rs         -- Error-to-HTTP mapping
  handlers/
    mod.rs
    health.rs      -- GET /health, GET /ready
    sync.rs        -- POST /sync, POST /index
    jobs.rs        -- POST /jobs/sync, POST /jobs/index, GET /jobs/{id}, POST /jobs/{id}/cancel
    search.rs      -- POST /search
    topics.rs      -- GET /topics, GET /topics/{path}/coverage, GET /topics/{path}/gaps
    duplicates.rs  -- GET /duplicates
```

## Internal Details

### Startup / Shutdown
- `main` configures tracing, loads settings, constructs `AppState`, builds router with middleware layers.
- Binds to `0.0.0.0:{port}` with graceful shutdown on SIGTERM/SIGINT.
- On shutdown: close database pool, close Qdrant, close Redis job manager.

### Authentication
- `X-API-Key` header checked if `ANKIATLAS_API_KEY` env var is set.
- If not configured, all requests pass through.
- Returns 401 on invalid/missing key.

### Sync Endpoint
- Validates source path exists and has `.anki2`/`.anki21` extension.
- Runs migrations if requested.
- Calls `sync_anki_collection`.
- Optionally runs `index_all_notes`.
- Returns `SyncResponse` with all stats.

### Job Endpoints
- `POST /jobs/sync` and `POST /jobs/index` return 202 with `JobAcceptedResponse`.
- `GET /jobs/{id}` returns `JobStatusResponse`.
- `POST /jobs/{id}/cancel` triggers cancellation.
- 503 when Redis is unavailable.

### Search Endpoint
- Builds `SearchFilters` from request.
- Calls `SearchService::search`.
- Enriches results with note details.
- Returns `SearchResponse` with stats and filter metadata.

### Error Response Format
```json
{
    "error": "ErrorTypeName",
    "message": "Human-readable message",
    "path": "/api/endpoint"
}
```

## Acceptance Criteria
- [ ] `GET /health` returns 200 with status "healthy" and version
- [ ] `GET /ready` checks PostgreSQL, Qdrant, and Redis availability
- [ ] `POST /sync` validates source path and extension
- [ ] `POST /sync` returns 400 for missing collection file
- [ ] `POST /sync` returns SyncResponse with all upsert counts
- [ ] `POST /index` returns IndexResponse with processed/embedded/skipped counts
- [ ] `POST /jobs/sync` returns 202 with job_id and poll_url
- [ ] `POST /jobs/index` returns 202 with job_id and poll_url
- [ ] `GET /jobs/{id}` returns 404 for unknown job
- [ ] `GET /jobs/{id}` returns JobStatusResponse with progress
- [ ] `POST /jobs/{id}/cancel` returns updated status
- [ ] `POST /search` accepts query with filters and returns results
- [ ] `POST /search` includes stats (semantic_only, fts_only, both)
- [ ] `GET /topics/{path}/coverage` returns coverage metrics or 404
- [ ] `GET /topics/{path}/gaps` returns gap analysis with missing/undercovered counts
- [ ] `GET /duplicates` accepts threshold, max_clusters query params
- [ ] API key middleware blocks requests when key is configured but missing/wrong
- [ ] API key middleware passes through when no key is configured
- [ ] Correlation ID middleware sets X-Request-ID in response
- [ ] Domain errors map to correct HTTP status codes (404, 409, 503, 500)
- [ ] Error responses include error type, message, and path
- [ ] Server shuts down gracefully on SIGTERM
- [ ] All request/response types derive Serialize/Deserialize correctly
- [ ] `make check` equivalent passes (clippy, fmt, test)

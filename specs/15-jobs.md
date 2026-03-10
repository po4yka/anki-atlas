# Spec: crate `jobs`

## Source Reference
Current Rust implementation: `crates/jobs/`
Historical rewrite input: `packages/jobs/` (service.py, tasks.py)

## Purpose
Async background job management backed by Redis. Provides job enqueueing, status tracking, progress updates, cancellation, and retry logic. Jobs are persisted as JSON in Redis with TTL-based expiration. The task functions (`job_sync`, `job_index`) execute the actual work, updating job status at each phase. Uses `rustis` (async Redis client) instead of arq.

## Dependencies
```toml
[dependencies]
common = { path = "../common" }
async-trait = "0.1"
rustis = { version = "0.13", features = ["tokio-runtime"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tokio = { version = "1", features = ["time", "sync"] }
tracing = "0.1"
uuid = { version = "1", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
url = "2"

[dev-dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros", "test-util"] }
mockall = "0.13"
```

## Public API

### Error (`src/error.rs`)

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JobError {
    #[error("job backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("job not found: {0}")]
    NotFound(String),

    #[error("job already in terminal state: {status}")]
    TerminalState { job_id: String, status: String },

    #[error("Redis error: {0}")]
    Redis(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("task execution error: {0}")]
    TaskExecution(String),
}
```

### Types (`src/types.rs`)

```rust
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Job type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, EnumString, Display)]
#[strum(serialize_all = "lowercase")]
pub enum JobType {
    Sync,
    Index,
}

/// Job status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, EnumString, Display)]
#[strum(serialize_all = "lowercase")]
pub enum JobStatus {
    Queued,
    Scheduled,
    Running,
    Retrying,
    Succeeded,
    Failed,
    CancelRequested,
    Cancelled,
}

impl JobStatus {
    /// True if the job has reached a final state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

pub const JOB_KEY_PREFIX: &str = "ankiatlas:job:";

/// Persisted metadata for an async job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub job_id: String,
    pub job_type: JobType,
    pub status: JobStatus,
    pub payload: HashMap<String, serde_json::Value>,
    pub progress: f64,             // 0.0 - 100.0
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
```

### Redis Persistence (`src/persistence.rs`)

```rust
use crate::types::JobRecord;
use crate::error::JobError;

/// Build Redis key from job ID.
pub fn job_key(job_id: &str) -> String;

/// Persist a job record in Redis with TTL.
pub async fn save_job_record(
    client: &rustis::client::Client,
    record: &JobRecord,
    ttl_seconds: u64,
) -> Result<(), JobError>;

/// Load a job record from Redis.
pub async fn load_job_record(
    client: &rustis::client::Client,
    job_id: &str,
) -> Result<Option<JobRecord>, JobError>;
```

### Redis Connection (`src/connection.rs`)

```rust
use crate::error::JobError;

/// Redis connection configuration parsed from URL.
#[derive(Debug, Clone)]
pub struct RedisConfig {
    pub host: String,
    pub port: u16,
    pub database: u32,
    pub username: Option<String>,
    pub password: Option<String>,
    pub tls: bool,
}

/// Parse a Redis URL (redis:// or rediss://) into config.
pub fn parse_redis_url(redis_url: &str) -> Result<RedisConfig, JobError>;

/// Create a rustis Client from a Redis URL.
pub async fn create_redis_client(redis_url: &str) -> Result<rustis::client::Client, JobError>;
```

### Job Manager (`src/manager.rs`)

```rust
use crate::types::{JobRecord, JobType, JobStatus};
use crate::error::JobError;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Job manager trait for DI / testing.
#[async_trait::async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait JobManager: Send + Sync {
    /// Enqueue a sync job.
    async fn enqueue_sync_job(
        &self,
        payload: HashMap<String, serde_json::Value>,
        run_at: Option<DateTime<Utc>>,
    ) -> Result<JobRecord, JobError>;

    /// Enqueue an index job.
    async fn enqueue_index_job(
        &self,
        payload: HashMap<String, serde_json::Value>,
        run_at: Option<DateTime<Utc>>,
    ) -> Result<JobRecord, JobError>;

    /// Get current job metadata.
    async fn get_job(&self, job_id: &str) -> Result<Option<JobRecord>, JobError>;

    /// Request cancellation of a queued/running job.
    async fn cancel_job(&self, job_id: &str) -> Result<Option<JobRecord>, JobError>;

    /// Close the connection pool.
    async fn close(&self) -> Result<(), JobError>;
}

/// Redis-backed job manager.
pub struct RedisJobManager {
    client: rustis::client::Client,
    queue_name: String,
    max_retries: u32,
    result_ttl_seconds: u64,
}

impl RedisJobManager {
    pub async fn new(redis_url: &str, queue_name: &str, max_retries: u32, result_ttl_seconds: u64)
        -> Result<Self, JobError>;
}

// #[async_trait] impl JobManager for RedisJobManager { ... }
```

### Task Functions (`src/tasks.rs`)

```rust
use crate::types::JobRecord;
use crate::error::JobError;
use std::collections::HashMap;

/// Task execution context with Redis client reference.
pub struct TaskContext {
    pub redis: rustis::client::Client,
    pub attempt: u32,
}

/// Background task: sync Anki collection and optionally index vectors.
///
/// Phases: migrations (15%) -> sync to PostgreSQL (40%) -> index vectors (75%) -> done (100%).
/// Checks for cancellation between phases.
/// On retryable failure, sets status to "retrying" and returns error.
/// On terminal failure, sets status to "failed".
pub async fn job_sync(
    ctx: &TaskContext,
    job_id: &str,
    payload: &HashMap<String, serde_json::Value>,
) -> Result<HashMap<String, serde_json::Value>, JobError>;

/// Background task: index notes to vector store.
///
/// Phases: start (10%) -> index (50%) -> done (100%).
pub async fn job_index(
    ctx: &TaskContext,
    job_id: &str,
    payload: &HashMap<String, serde_json::Value>,
) -> Result<HashMap<String, serde_json::Value>, JobError>;
```

### Module root (`src/lib.rs`)

```rust
pub mod connection;
pub mod error;
pub mod manager;
pub mod persistence;
pub mod tasks;
pub mod types;

pub use error::JobError;
pub use manager::{JobManager, RedisJobManager};
pub use types::{JobRecord, JobStatus, JobType, JOB_KEY_PREFIX};
```

## Internal Details

### Job Enqueueing
1. Generate UUID v4 for `job_id`.
2. Determine initial status: `Scheduled` if `run_at > now`, else `Queued`.
3. Create `JobRecord` with `created_at = now`, `scheduled_for = run_at`.
4. Persist to Redis with TTL.
5. Push job ID to Redis list (queue) for worker consumption.

### Cancellation Logic
- If job is in terminal state (`Succeeded`, `Failed`, `Cancelled`), return as-is.
- If job is `Queued`, `Scheduled`, or `Retrying`: immediately set to `Cancelled`.
- If job is `Running`: set to `CancelRequested`, let task check and cancel.
- Task functions check `cancel_requested` flag between phases.

### Status Updates
- `_set_status` helper loads record, updates fields, clamps progress to 0-100, saves back.
- `finished=true` sets `finished_at = now` and `progress = 100.0`.

### Redis URL Parsing
- Supports `redis://` and `rediss://` (TLS) schemes.
- Extracts host, port (default 6379), database (from path), username, password.

### Retry Logic
- Tasks receive `attempt` count from context.
- On retryable error and `attempt < max_retries`: set status to `Retrying`, propagate error.
- On terminal failure or exhausted retries: set status to `Failed`.
- `FileNotFoundError` equivalent is never retryable.

## Acceptance Criteria
- [ ] `JobStatus::is_terminal` returns true for Succeeded, Failed, Cancelled
- [ ] `JobRecord` serializes to/from JSON correctly (roundtrip test)
- [ ] `parse_redis_url` parses `redis://host:port/db` correctly
- [ ] `parse_redis_url` parses `rediss://user:pass@host:port/db` with TLS flag
- [ ] `parse_redis_url` rejects non-redis schemes
- [ ] `job_key` produces `ankiatlas:job:{id}`
- [ ] `save_job_record` + `load_job_record` roundtrip (integration test with Redis)
- [ ] `enqueue_sync_job` creates record with correct initial status (Queued vs Scheduled)
- [ ] `enqueue_index_job` creates record with correct initial status
- [ ] `cancel_job` immediately cancels Queued jobs
- [ ] `cancel_job` sets CancelRequested for Running jobs
- [ ] `cancel_job` returns existing record for terminal-state jobs
- [ ] `job_sync` updates progress through phases (5% -> 15% -> 40% -> 75% -> 100%)
- [ ] `job_sync` checks cancellation between phases
- [ ] `job_sync` handles retryable vs non-retryable errors
- [ ] `job_index` updates progress through phases
- [ ] `MockJobManager` compiles and can be used in API/CLI tests
- [ ] All types are `Send + Sync` (compile-time assertion)
- [ ] `make check` equivalent passes (clippy, fmt, test)

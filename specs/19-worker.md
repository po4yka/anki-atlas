# Spec: crate `worker`

## Source Reference
Current Rust implementation: `bins/worker/`
Historical rewrite input: `apps/worker.py`

## Purpose
Tokio-based background worker that polls Redis for queued jobs and executes them. Replaces the Python arq worker. Runs `job_sync` and `job_index` task functions from the `jobs` crate. Implements a simple Redis-based job queue consumer with configurable concurrency, retry handling, and graceful shutdown.

## Dependencies
```toml
[dependencies]
common = { path = "../common" }
jobs = { path = "../jobs" }
# Additional workspace crates needed by task functions:
# anki-sync, indexer, database

rustis = { version = "0.13", features = ["tokio-runtime"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "signal", "time", "sync"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
anyhow = "1"

[dev-dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros", "test-util"] }
```

## Public API

### Worker Configuration (`src/config.rs`)

```rust
use std::time::Duration;

/// Worker configuration, loaded from Settings.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Redis URL for job queue.
    pub redis_url: String,

    /// Name of the Redis list to poll for jobs.
    pub queue_name: String,

    /// Maximum concurrent job executions.
    pub max_concurrency: usize,     // default 4

    /// Maximum retry attempts per job.
    pub max_retries: u32,           // default 3

    /// Polling interval when queue is empty.
    pub poll_interval: Duration,    // default 1s

    /// Allow aborting running jobs on shutdown.
    pub allow_abort_on_shutdown: bool, // default true

    /// TTL for job results in Redis.
    pub result_ttl_seconds: u64,    // default from Settings
}

impl WorkerConfig {
    /// Build from common::config::Settings.
    pub fn from_settings(settings: &common::config::Settings) -> Self;
}
```

### Worker (`src/worker.rs`)

```rust
use crate::config::WorkerConfig;
use tokio::sync::Semaphore;
use std::sync::Arc;

/// Background job worker that polls Redis and dispatches tasks.
pub struct Worker {
    config: WorkerConfig,
    redis: rustis::client::Client,
    semaphore: Arc<Semaphore>,
    shutdown: tokio::sync::watch::Receiver<bool>,
}

impl Worker {
    /// Create a new worker with the given configuration.
    pub async fn new(config: WorkerConfig) -> anyhow::Result<Self>;

    /// Run the worker loop until shutdown signal is received.
    ///
    /// Poll loop:
    /// 1. BRPOP from queue with timeout (poll_interval).
    /// 2. Deserialize job envelope (job_id, task_name, payload).
    /// 3. Acquire semaphore permit (concurrency control).
    /// 4. Spawn tokio task to execute the job function.
    /// 5. On completion, update job status via jobs::persistence.
    pub async fn run(&self) -> anyhow::Result<()>;

    /// Request graceful shutdown.
    /// Waits for in-flight jobs to complete (up to timeout),
    /// then aborts remaining if allow_abort_on_shutdown is true.
    pub async fn shutdown(&self, timeout: std::time::Duration);
}
```

### Job Envelope (`src/envelope.rs`)

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Serialized job message pushed to the Redis queue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobEnvelope {
    pub job_id: String,
    pub task_name: String,          // "job_sync" or "job_index"
    pub payload: HashMap<String, serde_json::Value>,
}
```

### Task Dispatcher (`src/dispatcher.rs`)

```rust
use crate::envelope::JobEnvelope;
use jobs::tasks::TaskContext;

/// Dispatch a job envelope to the appropriate task function.
///
/// Maps task_name to:
/// - "job_sync"  -> jobs::tasks::job_sync
/// - "job_index" -> jobs::tasks::job_index
///
/// Unknown task names log a warning and set job status to "failed".
pub async fn dispatch(
    ctx: &TaskContext,
    envelope: &JobEnvelope,
) -> Result<HashMap<String, serde_json::Value>, jobs::JobError>;
```

### Entry Point (`src/main.rs`)

```rust
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Configure tracing (JSON to stderr, env-filter).
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .json()
        .init();

    // 2. Load settings.
    let settings = common::config::Settings::load()?;
    let config = WorkerConfig::from_settings(&settings);

    // 3. Create and run worker.
    let worker = Worker::new(config).await?;

    // 4. Spawn signal handler for graceful shutdown.
    tokio::select! {
        result = worker.run() => result,
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("shutdown signal received");
            worker.shutdown(std::time::Duration::from_secs(30)).await;
            Ok(())
        }
    }
}
```

### Module structure

```
src/
  main.rs          -- Entry point, signal handling
  config.rs        -- WorkerConfig
  worker.rs        -- Main worker loop
  envelope.rs      -- Job message format
  dispatcher.rs    -- Task name -> function mapping
```

## Internal Details

### Queue Protocol
- Jobs are enqueued by `RedisJobManager` as JSON-serialized `JobEnvelope` pushed to a Redis list (`LPUSH`).
- Worker consumes from the list with `BRPOP` (blocking pop with timeout).
- This is a simple FIFO queue. No priority or delayed execution at the queue level (scheduling is handled by `JobManager` which defers the push).

### Concurrency Control
- Uses `tokio::sync::Semaphore` with `max_concurrency` permits.
- Each job acquires a permit before execution, releases on completion.
- Prevents overloading the system with too many concurrent jobs.

### Retry Handling
- Worker tracks attempt count per job (from `JobRecord.attempts`).
- On task error: if `attempts < max_retries`, push job back to queue with incremented attempt.
- On terminal failure: set job status to `Failed` via `persistence::save_job_record`.
- Non-retryable errors (e.g., file not found) are never retried.

### Graceful Shutdown
1. Stop polling (break out of BRPOP loop).
2. Wait for in-flight semaphore permits to be released (all running jobs complete).
3. If timeout expires and `allow_abort_on_shutdown`: abort remaining tokio tasks.
4. Close Redis connection.

### Job Execution Flow
```
BRPOP queue -> deserialize JobEnvelope -> acquire semaphore
-> load JobRecord from Redis -> set status=Running
-> dispatch(task_name, payload)
-> on success: set status=Succeeded, store result
-> on retryable error: set status=Retrying, re-enqueue
-> on terminal error: set status=Failed, store error
-> release semaphore
```

### Logging
- Structured JSON logging via tracing.
- Each job execution logged with: job_id, task_name, attempt, duration, outcome.
- Queue polling logged at debug level.
- Errors logged at error level with full context.

## Acceptance Criteria
- [ ] `WorkerConfig::from_settings` correctly maps all Settings fields
- [ ] `JobEnvelope` serializes and deserializes correctly (roundtrip test)
- [ ] `dispatch` routes "job_sync" to `jobs::tasks::job_sync`
- [ ] `dispatch` routes "job_index" to `jobs::tasks::job_index`
- [ ] `dispatch` returns error for unknown task names
- [ ] `Worker::new` connects to Redis successfully
- [ ] `Worker::new` returns error on Redis connection failure
- [ ] Worker polls queue with BRPOP and processes jobs
- [ ] Worker respects max_concurrency via semaphore
- [ ] Worker handles empty queue (polls again after interval)
- [ ] Worker updates job status to Running before execution
- [ ] Worker updates job status to Succeeded on task success
- [ ] Worker updates job status to Failed on terminal error
- [ ] Worker re-enqueues job on retryable error with incremented attempt
- [ ] Worker stops re-enqueueing after max_retries exhausted
- [ ] Graceful shutdown waits for in-flight jobs
- [ ] Graceful shutdown aborts remaining after timeout (when configured)
- [ ] Worker shuts down on SIGTERM/SIGINT
- [ ] All types are `Send + Sync` (compile-time assertion)
- [ ] `make check` equivalent passes (clippy, fmt, test)

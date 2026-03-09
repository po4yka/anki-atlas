use crate::config::WorkerConfig;
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Trait abstracting Redis queue operations for testability.
///
/// Every external boundary behind a trait (project convention).
#[async_trait::async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait QueueBackend: Send + Sync {
    /// Blocking pop from the right of a Redis list. Returns None on timeout.
    async fn brpop(&self, key: &str, timeout: f64) -> anyhow::Result<Option<String>>;

    /// Push a value to the left of a Redis list (re-enqueue).
    async fn lpush(&self, key: &str, value: &str) -> anyhow::Result<()>;

    /// Load a job record by ID.
    async fn load_job_record(
        &self,
        job_id: &str,
    ) -> anyhow::Result<Option<jobs::types::JobRecord>>;

    /// Save a job record with TTL.
    async fn save_job_record(
        &self,
        record: &jobs::types::JobRecord,
        ttl_seconds: u64,
    ) -> anyhow::Result<()>;
}

/// Background job worker that polls Redis and dispatches tasks.
#[allow(dead_code)]
pub struct Worker<Q: QueueBackend> {
    config: WorkerConfig,
    queue: Arc<Q>,
    semaphore: Arc<Semaphore>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
}

impl<Q: QueueBackend + 'static> Worker<Q> {
    /// Create a new worker with the given configuration and queue backend.
    pub fn new(config: WorkerConfig, queue: Q) -> Self {
        // RED: stub - returns a struct but semaphore/shutdown not wired correctly
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        Self {
            semaphore: Arc::new(Semaphore::new(0)), // wrong: should be max_concurrency
            config,
            queue: Arc::new(queue),
            shutdown_tx,
            shutdown_rx,
        }
    }

    /// Run the worker loop until shutdown signal is received.
    pub async fn run(&self) -> anyhow::Result<()> {
        // RED: stub - returns immediately instead of polling
        Ok(())
    }

    /// Request graceful shutdown.
    pub async fn shutdown(&self, _timeout: std::time::Duration) {
        // RED: stub - does nothing
    }
}

#[cfg(test)]
mod tests;

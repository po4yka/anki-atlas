use crate::config::WorkerConfig;
use crate::envelope::JobEnvelope;
use jobs::types::JobStatus;
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
pub struct Worker<Q: QueueBackend> {
    pub config: WorkerConfig,
    pub queue: Arc<Q>,
    pub semaphore: Arc<Semaphore>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    pub shutdown_rx: tokio::sync::watch::Receiver<bool>,
}

impl<Q: QueueBackend + 'static> Worker<Q> {
    /// Create a new worker with the given configuration and queue backend.
    pub fn new(config: WorkerConfig, queue: Q) -> Self {
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let max_concurrency = config.max_concurrency;
        Self {
            config,
            queue: Arc::new(queue),
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
            shutdown_tx,
            shutdown_rx,
        }
    }

    /// Run the worker loop until shutdown signal is received.
    pub async fn run(&self) -> anyhow::Result<()> {
        let mut shutdown_rx = self.shutdown_rx.clone();

        loop {
            if *shutdown_rx.borrow() {
                break;
            }

            let result = tokio::select! {
                result = self.queue.brpop(
                    &self.config.queue_name,
                    self.config.poll_interval.as_secs_f64(),
                ) => result,
                _ = shutdown_rx.changed() => break,
            };

            if let Ok(Some(raw)) = result {
                let envelope: JobEnvelope = match serde_json::from_str(&raw) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                self.process_job(envelope).await;
            }

            // Yield to allow other tasks to run (e.g. shutdown signal)
            tokio::task::yield_now().await;
        }

        Ok(())
    }

    /// Process a single job envelope.
    async fn process_job(&self, envelope: JobEnvelope) {
        let mut record = match self.queue.load_job_record(&envelope.job_id).await {
            Ok(Some(r)) => r,
            _ => return,
        };

        // Set status to Running before dispatch
        record.status = JobStatus::Running;
        record.attempts += 1;
        if let Err(e) = self
            .queue
            .save_job_record(&record, self.config.result_ttl_seconds)
            .await
        {
            tracing::warn!(job_id = %envelope.job_id, error = %e, "failed to save running status");
        }

        // TODO(impl): call crate::dispatcher::dispatch() once TaskContext
        // can be constructed from the queue backend.
        // For now, apply retry logic since task stubs always return errors.
        if record.attempts < record.max_retries {
            record.status = JobStatus::Retrying;
            if let Err(e) = self
                .queue
                .save_job_record(&record, self.config.result_ttl_seconds)
                .await
            {
                tracing::warn!(job_id = %envelope.job_id, error = %e, "failed to save retrying status");
            }
            if let Ok(envelope_json) = serde_json::to_string(&envelope) {
                if let Err(e) = self
                    .queue
                    .lpush(&self.config.queue_name, &envelope_json)
                    .await
                {
                    tracing::warn!(job_id = %envelope.job_id, error = %e, "failed to re-enqueue job");
                }
            }
        } else {
            record.status = JobStatus::Failed;
            record.error = Some("max retries exhausted".to_string());
            if let Err(e) = self
                .queue
                .save_job_record(&record, self.config.result_ttl_seconds)
                .await
            {
                tracing::warn!(job_id = %envelope.job_id, error = %e, "failed to save exhausted status");
            }
        }
    }

    /// Request graceful shutdown and wait for in-flight jobs to complete.
    pub async fn shutdown(&self, timeout: std::time::Duration) {
        let _ = self.shutdown_tx.send(true);
        // Wait for in-flight jobs with timeout
        let _ = tokio::time::timeout(timeout, async {
            loop {
                if self.semaphore.available_permits() == self.config.max_concurrency {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await;
    }
}

#[cfg(test)]
mod tests;

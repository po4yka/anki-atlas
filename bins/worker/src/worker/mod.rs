use crate::config::WorkerConfig;
use crate::envelope::JobEnvelope;
use jobs::tasks::TaskContext;
use jobs::{JobError, JobRecord, JobResultData, JobStatus};
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
    async fn load_job_record(&self, job_id: &str)
    -> anyhow::Result<Option<jobs::types::JobRecord>>;

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
        let _permit = match self.semaphore.clone().acquire_owned().await {
            Ok(permit) => permit,
            Err(error) => {
                tracing::warn!(job_id = %envelope.job_id, error = %error, "worker semaphore closed");
                return;
            }
        };

        let mut record = match self.load_record(&envelope.job_id).await {
            Some(record) => record,
            None => return,
        };

        if record.status.is_terminal() {
            return;
        }

        if record.cancel_requested || matches!(record.status, JobStatus::CancelRequested) {
            self.finish_cancelled(&mut record).await;
            return;
        }

        if record.scheduled_for.is_some() {
            self.finish_failed(
                &mut record,
                JobError::Unsupported("scheduled jobs are not supported yet".to_string()),
            )
            .await;
            return;
        }

        record.status = JobStatus::Running;
        record.attempts += 1;
        if record.started_at.is_none() {
            record.started_at = Some(chrono::Utc::now());
        }
        record.message = Some("job started".to_string());

        if !self.save_record(&record, "save running status").await {
            return;
        }

        let ctx = TaskContext {
            attempt: record.attempts,
        };

        match crate::dispatcher::dispatch(&ctx, &envelope).await {
            Ok(result) => self.finish_succeeded(&mut record, result).await,
            Err(error) => self.finish_error(&mut record, &envelope, error).await,
        }
    }

    async fn load_record(&self, job_id: &str) -> Option<JobRecord> {
        match self.queue.load_job_record(job_id).await {
            Ok(Some(record)) => Some(record),
            Ok(None) => None,
            Err(error) => {
                tracing::warn!(job_id = %job_id, error = %error, "failed to load job record");
                None
            }
        }
    }

    async fn save_record(&self, record: &JobRecord, action: &str) -> bool {
        match self
            .queue
            .save_job_record(record, self.config.result_ttl_seconds)
            .await
        {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(job_id = %record.job_id, error = %error, action, "worker persistence failed");
                false
            }
        }
    }

    async fn finish_cancelled(&self, record: &mut JobRecord) {
        record.status = JobStatus::Cancelled;
        record.progress = 1.0;
        record.finished_at = Some(chrono::Utc::now());
        record.message = Some("job cancelled before execution".to_string());
        record.error = None;
        self.save_record(record, "save cancelled status").await;
    }

    async fn finish_succeeded(&self, record: &mut JobRecord, result: JobResultData) {
        record.status = JobStatus::Succeeded;
        record.progress = 1.0;
        record.finished_at = Some(chrono::Utc::now());
        record.message = Some("job completed".to_string());
        record.result = Some(result);
        record.error = None;
        self.save_record(record, "save succeeded status").await;
    }

    async fn finish_error(&self, record: &mut JobRecord, envelope: &JobEnvelope, error: JobError) {
        let error_message = error.to_string();

        if error.is_retryable() && record.attempts < record.max_retries && !record.cancel_requested
        {
            record.status = JobStatus::Retrying;
            record.message = Some("retrying after transient worker failure".to_string());
            record.error = Some(error_message);

            if self.save_record(record, "save retrying status").await {
                self.reenqueue(envelope).await;
            }
            return;
        }

        self.finish_failed(record, error).await;
    }

    async fn finish_failed(&self, record: &mut JobRecord, error: JobError) {
        record.status = JobStatus::Failed;
        record.progress = 1.0;
        record.finished_at = Some(chrono::Utc::now());
        record.message = Some("job failed".to_string());
        record.error = Some(error.to_string());
        self.save_record(record, "save failed status").await;
    }

    async fn reenqueue(&self, envelope: &JobEnvelope) {
        let envelope_json = match serde_json::to_string(envelope) {
            Ok(envelope_json) => envelope_json,
            Err(error) => {
                tracing::warn!(job_id = %envelope.job_id, error = %error, "failed to serialize job envelope");
                return;
            }
        };

        if let Err(error) = self
            .queue
            .lpush(&self.config.queue_name, &envelope_json)
            .await
        {
            tracing::warn!(job_id = %envelope.job_id, error = %error, "failed to re-enqueue job");
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

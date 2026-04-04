use sqlx::PgPool;

use crate::error::JobError;
use crate::types::{IndexJobPayload, JobPayload, JobRecord, JobStatus, JobType, SyncJobPayload};
use chrono::{DateTime, Utc};
use tracing::instrument;

/// Job manager trait for DI / testing.
#[async_trait::async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait JobManager: Send + Sync {
    /// Enqueue a sync job.
    async fn enqueue_sync_job(
        &self,
        payload: SyncJobPayload,
        run_at: Option<DateTime<Utc>>,
    ) -> Result<JobRecord, JobError>;

    /// Enqueue an index job.
    async fn enqueue_index_job(
        &self,
        payload: IndexJobPayload,
        run_at: Option<DateTime<Utc>>,
    ) -> Result<JobRecord, JobError>;

    /// Get current job metadata.
    async fn get_job(&self, job_id: &str) -> Result<JobRecord, JobError>;

    /// Request cancellation of a queued/running job.
    async fn cancel_job(&self, job_id: &str) -> Result<JobRecord, JobError>;

    /// Close the connection pool.
    async fn close(&self) -> Result<(), JobError>;
}

/// PostgreSQL-backed job manager.
///
/// Uses the `job_queue` table for both queue semantics and job state storage.
/// Replaces the previous Redis-backed implementation with zero external dependencies
/// beyond the existing PostgreSQL database.
pub struct PgJobManager {
    pool: PgPool,
    max_retries: u32,
    result_ttl_seconds: u64,
}

impl PgJobManager {
    /// Create a new PostgreSQL job manager.
    ///
    /// Automatically creates the `job_queue` table if it doesn't exist.
    pub async fn new(
        pool: PgPool,
        max_retries: u32,
        result_ttl_seconds: u64,
    ) -> Result<Self, JobError> {
        crate::persistence::ensure_schema(&pool).await?;
        Ok(Self {
            pool,
            max_retries,
            result_ttl_seconds,
        })
    }

    /// Access the underlying connection pool (used by the worker).
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    fn create_record(&self, job_type: JobType, payload: JobPayload) -> JobRecord {
        let now = Utc::now();
        JobRecord {
            job_id: uuid::Uuid::new_v4().to_string(),
            job_type,
            status: JobStatus::Queued,
            payload,
            progress: 0.0,
            message: None,
            attempts: 0,
            max_retries: self.max_retries,
            cancel_requested: false,
            created_at: Some(now),
            scheduled_for: None,
            started_at: None,
            finished_at: None,
            result: None,
            error: None,
        }
    }

    async fn enqueue(&self, record: &JobRecord) -> Result<(), JobError> {
        crate::persistence::save_job_record(&self.pool, record, self.result_ttl_seconds).await
    }
}

#[async_trait::async_trait]
impl JobManager for PgJobManager {
    #[instrument(skip(self))]
    async fn enqueue_sync_job(
        &self,
        payload: SyncJobPayload,
        run_at: Option<DateTime<Utc>>,
    ) -> Result<JobRecord, JobError> {
        if run_at.is_some() {
            return Err(JobError::Unsupported(
                "scheduled jobs are not supported yet".to_string(),
            ));
        }

        let record = self.create_record(JobType::Sync, JobPayload::Sync(payload));
        self.enqueue(&record).await?;
        Ok(record)
    }

    #[instrument(skip(self))]
    async fn enqueue_index_job(
        &self,
        payload: IndexJobPayload,
        run_at: Option<DateTime<Utc>>,
    ) -> Result<JobRecord, JobError> {
        if run_at.is_some() {
            return Err(JobError::Unsupported(
                "scheduled jobs are not supported yet".to_string(),
            ));
        }

        let record = self.create_record(JobType::Index, JobPayload::Index(payload));
        self.enqueue(&record).await?;
        Ok(record)
    }

    #[instrument(skip(self))]
    async fn get_job(&self, job_id: &str) -> Result<JobRecord, JobError> {
        crate::persistence::load_job_record(&self.pool, job_id)
            .await?
            .ok_or_else(|| JobError::NotFound(job_id.to_string()))
    }

    #[instrument(skip(self))]
    async fn cancel_job(&self, job_id: &str) -> Result<JobRecord, JobError> {
        let mut record = self.get_job(job_id).await?;
        if record.status.is_terminal() {
            return Err(JobError::TerminalState {
                job_id: job_id.to_string(),
                status: record.status.to_string(),
            });
        }
        record.cancel_requested = true;
        record.status = JobStatus::CancelRequested;
        crate::persistence::save_job_record(&self.pool, &record, self.result_ttl_seconds).await?;
        Ok(record)
    }

    #[instrument(skip(self))]
    async fn close(&self) -> Result<(), JobError> {
        // Cleanup expired jobs on close
        let deleted = crate::persistence::cleanup_expired(&self.pool).await?;
        if deleted > 0 {
            tracing::info!(deleted, "cleaned up expired job records");
        }
        Ok(())
    }
}

use crate::error::JobError;
use crate::types::{
    IndexJobPayload, JobEnvelope, JobPayload, JobRecord, JobStatus, JobType, SyncJobPayload,
};
use chrono::{DateTime, Utc};

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

/// Redis-backed job manager.
pub struct RedisJobManager {
    client: rustis::client::Client,
    queue_name: String,
    max_retries: u32,
    result_ttl_seconds: u64,
}

impl RedisJobManager {
    pub async fn new(
        redis_url: &str,
        queue_name: &str,
        max_retries: u32,
        result_ttl_seconds: u64,
    ) -> Result<Self, JobError> {
        let client = crate::connection::create_redis_client(redis_url).await?;
        Ok(Self {
            client,
            queue_name: queue_name.to_string(),
            max_retries,
            result_ttl_seconds,
        })
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
        use rustis::commands::ListCommands;

        crate::persistence::save_job_record(&self.client, record, self.result_ttl_seconds).await?;

        let envelope = JobEnvelope::from(record);
        let json =
            serde_json::to_string(&envelope).map_err(|e| JobError::Serialization(e.to_string()))?;
        let _: usize = self
            .client
            .lpush(&self.queue_name, &json)
            .await
            .map_err(|e| JobError::Redis(e.to_string()))?;

        Ok(())
    }
}

#[async_trait::async_trait]
impl JobManager for RedisJobManager {
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

    async fn get_job(&self, job_id: &str) -> Result<JobRecord, JobError> {
        crate::persistence::load_job_record(&self.client, job_id)
            .await?
            .ok_or_else(|| JobError::NotFound(job_id.to_string()))
    }

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
        crate::persistence::save_job_record(&self.client, &record, self.result_ttl_seconds).await?;
        Ok(record)
    }

    async fn close(&self) -> Result<(), JobError> {
        Ok(())
    }
}

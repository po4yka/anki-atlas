use crate::error::JobError;
use crate::types::{JobRecord, JobStatus, JobType};
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

    fn create_record(
        &self,
        job_type: JobType,
        payload: HashMap<String, serde_json::Value>,
        run_at: Option<DateTime<Utc>>,
    ) -> JobRecord {
        let now = Utc::now();
        let status = if run_at.is_some() {
            JobStatus::Scheduled
        } else {
            JobStatus::Queued
        };
        JobRecord {
            job_id: uuid::Uuid::new_v4().to_string(),
            job_type,
            status,
            payload,
            progress: 0.0,
            message: None,
            attempts: 0,
            max_retries: self.max_retries,
            cancel_requested: false,
            created_at: Some(now),
            scheduled_for: run_at,
            started_at: None,
            finished_at: None,
            result: None,
            error: None,
        }
    }

    async fn enqueue(&self, record: &JobRecord) -> Result<(), JobError> {
        use rustis::commands::ListCommands;

        crate::persistence::save_job_record(&self.client, record, self.result_ttl_seconds).await?;

        let json =
            serde_json::to_string(record).map_err(|e| JobError::Serialization(e.to_string()))?;
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
        payload: HashMap<String, serde_json::Value>,
        run_at: Option<DateTime<Utc>>,
    ) -> Result<JobRecord, JobError> {
        let record = self.create_record(JobType::Sync, payload, run_at);
        self.enqueue(&record).await?;
        Ok(record)
    }

    async fn enqueue_index_job(
        &self,
        payload: HashMap<String, serde_json::Value>,
        run_at: Option<DateTime<Utc>>,
    ) -> Result<JobRecord, JobError> {
        let record = self.create_record(JobType::Index, payload, run_at);
        self.enqueue(&record).await?;
        Ok(record)
    }

    async fn get_job(&self, job_id: &str) -> Result<Option<JobRecord>, JobError> {
        crate::persistence::load_job_record(&self.client, job_id).await
    }

    async fn cancel_job(&self, job_id: &str) -> Result<Option<JobRecord>, JobError> {
        let record = crate::persistence::load_job_record(&self.client, job_id).await?;
        match record {
            Some(mut rec) => {
                if rec.status.is_terminal() {
                    return Err(JobError::TerminalState {
                        job_id: job_id.to_string(),
                        status: rec.status.to_string(),
                    });
                }
                rec.cancel_requested = true;
                rec.status = JobStatus::CancelRequested;
                crate::persistence::save_job_record(
                    &self.client,
                    &rec,
                    self.result_ttl_seconds,
                )
                .await?;
                Ok(Some(rec))
            }
            None => Err(JobError::NotFound(job_id.to_string())),
        }
    }

    async fn close(&self) -> Result<(), JobError> {
        Ok(())
    }
}

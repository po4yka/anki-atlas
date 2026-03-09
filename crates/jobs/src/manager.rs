use crate::error::JobError;
use crate::types::JobRecord;
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
#[allow(dead_code)]
pub struct RedisJobManager {
    client: rustis::client::Client,
    queue_name: String,
    max_retries: u32,
    result_ttl_seconds: u64,
}

impl RedisJobManager {
    pub async fn new(
        _redis_url: &str,
        _queue_name: &str,
        _max_retries: u32,
        _result_ttl_seconds: u64,
    ) -> Result<Self, JobError> {
        // TODO(impl): implement
        Err(JobError::BackendUnavailable("not implemented".to_string()))
    }
}

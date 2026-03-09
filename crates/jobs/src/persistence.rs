use crate::error::JobError;
use crate::types::{JobRecord, JOB_KEY_PREFIX};

/// Build Redis key from job ID.
pub fn job_key(job_id: &str) -> String {
    // TODO(impl): implement
    let _ = JOB_KEY_PREFIX;
    String::new()
}

/// Persist a job record in Redis with TTL.
pub async fn save_job_record(
    _client: &rustis::client::Client,
    _record: &JobRecord,
    _ttl_seconds: u64,
) -> Result<(), JobError> {
    // TODO(impl): implement
    Err(JobError::Redis("not implemented".to_string()))
}

/// Load a job record from Redis.
pub async fn load_job_record(
    _client: &rustis::client::Client,
    _job_id: &str,
) -> Result<Option<JobRecord>, JobError> {
    // TODO(impl): implement
    Err(JobError::Redis("not implemented".to_string()))
}

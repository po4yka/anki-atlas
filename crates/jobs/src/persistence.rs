use crate::error::JobError;
use crate::types::{JOB_KEY_PREFIX, JobRecord};
use rustis::commands::StringCommands;

/// Build Redis key from job ID.
pub fn job_key(job_id: &str) -> String {
    format!("{JOB_KEY_PREFIX}{job_id}")
}

/// Persist a job record in Redis with TTL.
pub async fn save_job_record(
    client: &rustis::client::Client,
    record: &JobRecord,
    ttl_seconds: u64,
) -> Result<(), JobError> {
    let key = job_key(&record.job_id);
    let json = serde_json::to_string(record).map_err(|e| JobError::Serialization(e.to_string()))?;
    client
        .setex(&key, ttl_seconds, &json)
        .await
        .map_err(|e| JobError::Redis(e.to_string()))?;
    Ok(())
}

/// Load a job record from Redis.
pub async fn load_job_record(
    client: &rustis::client::Client,
    job_id: &str,
) -> Result<Option<JobRecord>, JobError> {
    let key = job_key(job_id);
    let raw: Option<String> = client
        .get(&key)
        .await
        .map_err(|e| JobError::Redis(e.to_string()))?;
    match raw {
        Some(s) => {
            let record =
                serde_json::from_str(&s).map_err(|e| JobError::Serialization(e.to_string()))?;
            Ok(Some(record))
        }
        None => Ok(None),
    }
}

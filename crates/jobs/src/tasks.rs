use crate::error::JobError;
use std::collections::HashMap;

/// Task execution context with Redis client reference.
pub struct TaskContext {
    pub redis: rustis::client::Client,
    pub attempt: u32,
}

/// Background task: sync Anki collection and optionally index vectors.
///
/// Phases: migrations (15%) -> sync to PostgreSQL (40%) -> index vectors (75%) -> done (100%).
/// Checks for cancellation between phases.
/// On retryable failure, sets status to "retrying" and returns error.
/// On terminal failure, sets status to "failed".
pub async fn job_sync(
    _ctx: &TaskContext,
    _job_id: &str,
    _payload: &HashMap<String, serde_json::Value>,
) -> Result<HashMap<String, serde_json::Value>, JobError> {
    // TODO(impl): implement
    Err(JobError::TaskExecution("not implemented".to_string()))
}

/// Background task: index notes to vector store.
///
/// Phases: start (10%) -> index (50%) -> done (100%).
pub async fn job_index(
    _ctx: &TaskContext,
    _job_id: &str,
    _payload: &HashMap<String, serde_json::Value>,
) -> Result<HashMap<String, serde_json::Value>, JobError> {
    // TODO(impl): implement
    Err(JobError::TaskExecution("not implemented".to_string()))
}

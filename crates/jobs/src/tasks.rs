use crate::error::JobError;
use crate::types::{IndexJobPayload, IndexJobResult, SyncJobPayload, SyncJobResult};

/// Task execution context for one job attempt.
#[derive(Debug, Clone, Copy)]
pub struct TaskContext {
    pub attempt: u32,
}

/// Background task: sync Anki collection and optionally index vectors.
///
/// The async job runtime is wired end-to-end, but task execution is still pending.
pub async fn job_sync(
    _ctx: &TaskContext,
    _job_id: &str,
    _payload: &SyncJobPayload,
) -> Result<SyncJobResult, JobError> {
    Err(JobError::Unsupported(
        "sync job execution is not implemented yet".to_string(),
    ))
}

/// Background task: index notes to vector store.
///
/// The async job runtime is wired end-to-end, but task execution is still pending.
pub async fn job_index(
    _ctx: &TaskContext,
    _job_id: &str,
    _payload: &IndexJobPayload,
) -> Result<IndexJobResult, JobError> {
    Err(JobError::Unsupported(
        "index job execution is not implemented yet".to_string(),
    ))
}

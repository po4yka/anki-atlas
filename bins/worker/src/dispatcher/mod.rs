use crate::envelope::JobEnvelope;
use jobs::error::JobError;
use jobs::tasks::TaskContext;
use std::collections::HashMap;

/// Dispatch a job envelope to the appropriate task function.
///
/// Maps task_name to:
/// - "job_sync"  -> jobs::tasks::job_sync
/// - "job_index" -> jobs::tasks::job_index
///
/// Unknown task names return an error.
pub async fn dispatch(
    _ctx: &TaskContext,
    _envelope: &JobEnvelope,
) -> Result<HashMap<String, serde_json::Value>, JobError> {
    todo!()
}

#[cfg(test)]
mod tests;

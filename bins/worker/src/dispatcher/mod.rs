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
    ctx: &TaskContext,
    envelope: &JobEnvelope,
) -> Result<HashMap<String, serde_json::Value>, JobError> {
    match envelope.task_name.as_str() {
        "job_sync" => {
            jobs::tasks::job_sync(ctx, &envelope.job_id, &envelope.payload).await
        }
        "job_index" => {
            jobs::tasks::job_index(ctx, &envelope.job_id, &envelope.payload).await
        }
        other => Err(JobError::TaskExecution(format!(
            "unknown task: {other}"
        ))),
    }
}

#[cfg(test)]
mod tests;

use crate::envelope::JobEnvelope;
use jobs::error::JobError;
use jobs::tasks::TaskContext;
use jobs::types::JobType;
use std::collections::HashMap;

/// Dispatch a job envelope to the appropriate task function.
///
/// Routes based on `JobType` enum:
/// - `Sync`  -> jobs::tasks::job_sync
/// - `Index` -> jobs::tasks::job_index
pub async fn dispatch(
    ctx: &TaskContext,
    envelope: &JobEnvelope,
) -> Result<HashMap<String, serde_json::Value>, JobError> {
    match envelope.job_type {
        JobType::Sync => {
            jobs::tasks::job_sync(ctx, &envelope.job_id, &envelope.payload).await
        }
        JobType::Index => {
            jobs::tasks::job_index(ctx, &envelope.job_id, &envelope.payload).await
        }
    }
}

#[cfg(test)]
mod tests;

use crate::envelope::JobEnvelope;
use jobs::error::JobError;
use jobs::tasks::TaskContext;
use jobs::types::JobResultData;

/// Dispatch a job envelope to the appropriate task function.
///
/// Routes based on `JobType` enum:
/// - `Sync`  -> jobs::tasks::job_sync
/// - `Index` -> jobs::tasks::job_index
pub async fn dispatch(
    ctx: &TaskContext,
    envelope: &JobEnvelope,
) -> Result<JobResultData, JobError> {
    match &envelope.payload {
        jobs::types::JobPayload::Sync(payload) => {
            let result = jobs::tasks::job_sync(ctx, &envelope.job_id, payload).await?;
            Ok(JobResultData::Sync(result))
        }
        jobs::types::JobPayload::Index(payload) => {
            let result = jobs::tasks::job_index(ctx, &envelope.job_id, payload).await?;
            Ok(JobResultData::Index(result))
        }
    }
}

#[cfg(test)]
mod tests;

pub mod connection;
pub mod error;
pub mod manager;
pub mod persistence;
pub mod tasks;
pub mod types;

pub use error::JobError;
pub use manager::{JobManager, PgJobManager};
pub use types::{
    IndexJobPayload, IndexJobResult, JobEnvelope, JobPayload, JobRecord, JobResultData, JobStatus,
    JobType, SyncJobPayload, SyncJobResult,
};

#[cfg(test)]
mod tests;

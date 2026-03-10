use thiserror::Error;

#[derive(Debug, Error)]
pub enum JobError {
    #[error("job backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("job not found: {0}")]
    NotFound(String),

    #[error("job already in terminal state: {status}")]
    TerminalState { job_id: String, status: String },

    #[error("unsupported jobs feature: {0}")]
    Unsupported(String),

    #[error("Redis error: {0}")]
    Redis(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("task execution error: {0}")]
    TaskExecution(String),
}

impl JobError {
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::BackendUnavailable(_) | Self::Redis(_))
    }
}

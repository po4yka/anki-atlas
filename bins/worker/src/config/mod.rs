use std::time::Duration;

use common::config::JobSettings;

/// Worker configuration, loaded from Settings.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerConfig {
    /// PostgreSQL URL for job queue (shared with application database).
    pub postgres_url: String,

    /// Logical queue name (used for logging, not for table routing).
    pub queue_name: String,

    /// Maximum concurrent job executions.
    pub max_concurrency: usize,

    /// Maximum retry attempts per job.
    pub max_retries: u32,

    /// Polling interval when queue is empty.
    pub poll_interval: Duration,

    /// Allow aborting running jobs on shutdown.
    pub allow_abort_on_shutdown: bool,

    /// TTL for completed job records (seconds).
    pub result_ttl_seconds: u64,
}

impl WorkerConfig {
    /// Build from job-specific runtime settings.
    pub fn from_job_settings(settings: &JobSettings) -> Self {
        Self {
            postgres_url: settings.postgres_url.clone(),
            queue_name: settings.queue_name.clone(),
            max_concurrency: 4,
            max_retries: settings.max_retries,
            poll_interval: Duration::from_secs(1),
            allow_abort_on_shutdown: true,
            result_ttl_seconds: u64::from(settings.result_ttl_seconds),
        }
    }
}

#[cfg(test)]
mod tests;

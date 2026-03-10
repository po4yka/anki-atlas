use std::time::Duration;

use common::config::JobSettings;

/// Worker configuration, loaded from Settings.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerConfig {
    /// Redis URL for job queue.
    pub redis_url: String,

    /// Name of the Redis list to poll for jobs.
    pub queue_name: String,

    /// Maximum concurrent job executions.
    pub max_concurrency: usize,

    /// Maximum retry attempts per job.
    pub max_retries: u32,

    /// Polling interval when queue is empty.
    pub poll_interval: Duration,

    /// Allow aborting running jobs on shutdown.
    pub allow_abort_on_shutdown: bool,

    /// TTL for job results in Redis.
    pub result_ttl_seconds: u64,
}

impl WorkerConfig {
    /// Build from job-specific runtime settings.
    pub fn from_job_settings(settings: &JobSettings) -> Self {
        Self {
            redis_url: settings.redis_url.clone(),
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

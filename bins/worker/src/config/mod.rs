use std::time::Duration;

/// Worker configuration, loaded from Settings.
#[derive(Debug, Clone)]
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
    /// Build from common::config::Settings.
    pub fn from_settings(_settings: &common::config::Settings) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod tests;

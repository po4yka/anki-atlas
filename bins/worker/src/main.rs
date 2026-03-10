use std::env;

pub mod config;
pub mod dispatcher;
pub mod envelope;
pub mod worker;

use config::WorkerConfig;
use worker::{QueueBackend, Worker};

use common::logging::{LoggingConfig, init_global_logging};
use jobs::JobRecord;

/// Redis-backed queue backend for production use.
struct RedisQueueBackend {
    client: rustis::client::Client,
}

fn ensure_worker_runtime_enabled() -> anyhow::Result<()> {
    let enabled = env::var("ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false);

    if enabled {
        return Ok(());
    }

    anyhow::bail!(
        "anki-atlas-worker is disabled until background job task execution is implemented; \
set ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1 only for development and test runs"
    )
}

impl RedisQueueBackend {
    async fn connect(redis_url: &str) -> anyhow::Result<Self> {
        let client = jobs::connection::create_redis_client(redis_url)
            .await
            .map_err(anyhow::Error::from)?;
        Ok(Self { client })
    }
}

#[async_trait::async_trait]
impl QueueBackend for RedisQueueBackend {
    async fn brpop(&self, key: &str, timeout: f64) -> anyhow::Result<Option<String>> {
        use rustis::commands::BlockingCommands;
        let result: Option<(String, String)> = self.client.brpop(key, timeout).await?;
        Ok(result.map(|(_, value)| value))
    }

    async fn lpush(&self, key: &str, value: &str) -> anyhow::Result<()> {
        use rustis::commands::ListCommands;
        let _: usize = self.client.lpush(key, value).await?;
        Ok(())
    }

    async fn load_job_record(&self, job_id: &str) -> anyhow::Result<Option<JobRecord>> {
        jobs::persistence::load_job_record(&self.client, job_id)
            .await
            .map_err(anyhow::Error::from)
    }

    async fn save_job_record(&self, record: &JobRecord, ttl_seconds: u64) -> anyhow::Result<()> {
        jobs::persistence::save_job_record(&self.client, record, ttl_seconds)
            .await
            .map_err(anyhow::Error::from)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let settings = common::config::Settings::load()?;
    let api_settings = settings.api();
    let job_settings = settings.jobs();

    init_global_logging(&LoggingConfig {
        debug: api_settings.debug,
        json_output: false,
    })?;

    let config = WorkerConfig::from_job_settings(&job_settings);

    ensure_worker_runtime_enabled()?;

    tracing::info!(queue = %config.queue_name, concurrency = config.max_concurrency, "starting worker");

    let backend = RedisQueueBackend::connect(&config.redis_url).await?;
    let worker = Worker::new(config, backend);

    tokio::select! {
        result = worker.run() => result?,
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("shutdown signal received");
            worker.shutdown(std::time::Duration::from_secs(30)).await;
        }
    }

    Ok(())
}

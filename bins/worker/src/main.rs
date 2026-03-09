pub mod config;
pub mod dispatcher;
pub mod envelope;
pub mod worker;

use config::WorkerConfig;
use worker::{QueueBackend, Worker};

use jobs::types::{JobRecord, JOB_KEY_PREFIX};

/// Redis-backed queue backend for production use.
struct RedisQueueBackend {
    client: rustis::client::Client,
}

impl RedisQueueBackend {
    async fn connect(redis_url: &str) -> anyhow::Result<Self> {
        let client = rustis::client::Client::connect(redis_url).await?;
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
        use rustis::commands::StringCommands;
        let key = format!("{JOB_KEY_PREFIX}{job_id}");
        let raw: Option<String> = self.client.get(&key).await?;
        match raw {
            Some(s) => Ok(Some(serde_json::from_str(&s)?)),
            None => Ok(None),
        }
    }

    async fn save_job_record(&self, record: &JobRecord, ttl_seconds: u64) -> anyhow::Result<()> {
        use rustis::commands::StringCommands;
        let key = format!("{JOB_KEY_PREFIX}{}", record.job_id);
        let json = serde_json::to_string(record)?;
        self.client
            .set_with_options(
                &key,
                &json,
                rustis::commands::SetCondition::None,
                rustis::commands::SetExpiration::Ex(ttl_seconds),
                false,
            )
            .await?;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let settings = common::config::Settings::load()?;
    let config = WorkerConfig::from_settings(&settings);

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

use std::env;

pub mod config;
pub mod dispatcher;
pub mod envelope;
pub mod tasks;
pub mod worker;

use config::WorkerConfig;
use worker::{QueueBackend, Worker};

use common::logging::{LoggingConfig, init_global_logging};
use jobs::JobRecord;

/// PostgreSQL-backed queue backend for production use.
///
/// Uses `SELECT ... FOR UPDATE SKIP LOCKED` for reliable job dequeuing
/// and standard INSERT/UPDATE for job state persistence.
struct PgQueueBackend {
    pool: sqlx::PgPool,
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

impl PgQueueBackend {
    async fn connect(postgres_url: &str) -> anyhow::Result<Self> {
        let pool = jobs::connection::create_job_pool(postgres_url)
            .await
            .map_err(anyhow::Error::from)?;
        // Ensure schema exists
        jobs::persistence::ensure_schema(&pool)
            .await
            .map_err(anyhow::Error::from)?;
        Ok(Self { pool })
    }
}

impl QueueBackend for PgQueueBackend {
    async fn brpop(&self, _key: &str, timeout: f64) -> anyhow::Result<Option<String>> {
        // Try to pop a job using FOR UPDATE SKIP LOCKED
        match jobs::persistence::pop_next_job(&self.pool).await? {
            Some(record) => {
                // Convert to envelope JSON for the worker
                let envelope = jobs::JobEnvelope::from(&record);
                let json = serde_json::to_string(&envelope)?;
                Ok(Some(json))
            }
            None => {
                // No jobs available -- sleep for the timeout period
                tokio::time::sleep(std::time::Duration::from_secs_f64(timeout)).await;
                Ok(None)
            }
        }
    }

    async fn lpush(&self, _key: &str, value: &str) -> anyhow::Result<()> {
        // Re-enqueue: parse the envelope and set status back to queued
        let envelope: jobs::JobEnvelope = serde_json::from_str(value)?;
        let record = self.load_job_record(&envelope.job_id).await?;
        if let Some(record) = record {
            jobs::persistence::reenqueue_job(&self.pool, &record).await?;
        }
        Ok(())
    }

    async fn load_job_record(&self, job_id: &str) -> anyhow::Result<Option<JobRecord>> {
        jobs::persistence::load_job_record(&self.pool, job_id)
            .await
            .map_err(anyhow::Error::from)
    }

    async fn save_job_record(&self, record: &JobRecord, ttl_seconds: u64) -> anyhow::Result<()> {
        jobs::persistence::save_job_record(&self.pool, record, ttl_seconds)
            .await
            .map_err(anyhow::Error::from)
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
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

    let backend = PgQueueBackend::connect(&config.postgres_url).await?;
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

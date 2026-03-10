use std::sync::Arc;

use anki_atlas_api::router::build_router;
use anki_atlas_api::state::AppState;
use common::logging::{LoggingConfig, init_global_logging};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let settings = common::config::Settings::load()?;
    let api_settings = settings.api();
    let job_settings = settings.jobs();

    init_global_logging(&LoggingConfig {
        debug: api_settings.debug,
        json_output: false,
    })?;

    let bind_addr = format!("{}:{}", api_settings.host, api_settings.port);

    let job_manager = jobs::RedisJobManager::new(
        &job_settings.redis_url,
        &job_settings.queue_name,
        job_settings.max_retries,
        u64::from(job_settings.result_ttl_seconds),
    )
    .await?;

    let state = AppState {
        api: Arc::new(api_settings),
        job_manager: Arc::new(job_manager),
    };

    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    tracing::info!("API server listening on {bind_addr}");
    axum::serve(listener, app).await?;

    Ok(())
}

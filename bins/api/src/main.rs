use std::sync::Arc;

use anki_atlas_api::router::build_router;
use anki_atlas_api::state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let settings = common::config::Settings::load()?;
    let bind_addr = format!("{}:{}", settings.api_host, settings.api_port);

    let job_manager = jobs::RedisJobManager::new(
        &settings.redis_url,
        &settings.job_queue_name,
        settings.job_max_retries,
        u64::from(settings.job_result_ttl_seconds),
    )
    .await?;

    let state = AppState {
        settings: Arc::new(settings),
        job_manager: Arc::new(job_manager),
    };

    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    tracing::info!("API server listening on {bind_addr}");
    axum::serve(listener, app).await?;

    Ok(())
}

use anki_atlas_api::router::build_router;
use anki_atlas_api::services::{build_api_services, build_app_state};
use common::logging::{LoggingConfig, init_global_logging};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    let settings = common::config::Settings::load()?;
    let api_settings = settings.api();

    init_global_logging(&LoggingConfig {
        debug: api_settings.debug,
        json_output: false,
    })?;

    let bind_addr = format!("{}:{}", api_settings.host, api_settings.port);
    let services = build_api_services(&settings).await?;
    let state = build_app_state(api_settings, services);

    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    tracing::info!("API server listening on {bind_addr}");
    axum::serve(listener, app).await?;

    Ok(())
}

use axum::Json;
use serde_json::{json, Value};
use tracing::instrument;

/// Returns 200 with service status and version.
#[instrument]
pub async fn health() -> Json<Value> {
    Json(json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// Returns 200 when the service is ready to accept traffic.
#[instrument]
pub async fn ready() -> Json<Value> {
    Json(json!({
        "status": "ready",
    }))
}

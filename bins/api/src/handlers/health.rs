use axum::Json;
use tracing::instrument;

use crate::schemas::{HealthResponse, ReadyResponse};

/// Returns 200 with service status and version.
#[instrument]
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy",
        version: env!("CARGO_PKG_VERSION"),
    })
}

/// Returns 200 when the service is ready to accept traffic.
#[instrument]
pub async fn ready() -> Json<ReadyResponse> {
    Json(ReadyResponse { status: "ready" })
}

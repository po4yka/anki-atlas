use axum::response::Response;
use tracing::instrument;

use crate::error::AppError;

/// Returns information about the current search index state.
#[instrument]
pub async fn index_info() -> Result<Response, AppError> {
    Err(super::unwired_surface(
        "the /index/info endpoint",
        "wire it to the real indexer status source before exposing it",
    ))
}

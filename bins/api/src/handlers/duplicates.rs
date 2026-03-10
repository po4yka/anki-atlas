use axum::response::Response;
use tracing::instrument;

use crate::error::AppError;

/// Find duplicate note clusters based on similarity threshold.
#[instrument]
pub async fn find_duplicates() -> Result<Response, AppError> {
    Err(super::unwired_surface(
        "the /duplicates endpoint",
        "wire it through the analytics service before advertising duplicate detection",
    ))
}

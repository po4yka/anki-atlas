use axum::Json;
use axum::response::Response;
use tracing::instrument;

use crate::error::AppError;
use crate::schemas::{SearchRequest, SearchResponse};

/// Hybrid search across indexed notes. Returns ranked results with scores.
#[instrument(skip_all)]
pub async fn search(Json(_req): Json<SearchRequest>) -> Result<Response, AppError> {
    let _response_shape: Option<SearchResponse> = None;
    Err(super::unwired_surface(
        "the /search endpoint",
        "wire the HTTP layer to the search service before exposing this route",
    ))
}

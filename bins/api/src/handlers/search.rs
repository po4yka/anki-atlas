use axum::Json;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use tracing::instrument;

use crate::error::AppError;
use crate::schemas::{ChunkSearchRequest, SearchRequest};
use crate::state::AppState;

#[instrument(skip(state, req))]
pub async fn search(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Response, AppError> {
    req.validate().map_err(AppError::bad_request)?;
    let result = state
        .services
        .search
        .search(&req)
        .await
        .map_err(AppError::from)?;

    Ok(axum::Json(result).into_response())
}

#[instrument(skip(state, req))]
pub async fn search_chunks(
    State(state): State<AppState>,
    Json(req): Json<ChunkSearchRequest>,
) -> Result<Response, AppError> {
    req.validate().map_err(AppError::bad_request)?;
    let result = state
        .services
        .search
        .search_chunks(&req)
        .await
        .map_err(AppError::from)?;

    Ok(axum::Json(result).into_response())
}

use axum::Json;
use axum::extract::{RawQuery, State};
use axum::response::{IntoResponse, Response};
use tracing::instrument;

use crate::error::AppError;
use crate::schemas::{DuplicatesQuery, DuplicatesResponse};
use crate::state::AppState;

#[instrument(skip(state))]
pub async fn duplicates(
    State(state): State<AppState>,
    RawQuery(raw_query): RawQuery,
) -> Result<Response, AppError> {
    let query =
        DuplicatesQuery::from_query_string(raw_query.as_deref()).map_err(AppError::bad_request)?;

    let (clusters, stats) = state
        .services
        .analytics
        .find_duplicates(
            query.threshold,
            query.max_clusters,
            query.deck_filter,
            query.tag_filter,
        )
        .await
        .map_err(AppError::from)?;

    Ok(Json(DuplicatesResponse {
        clusters: clusters.into_iter().map(Into::into).collect(),
        stats: stats.into(),
    })
    .into_response())
}

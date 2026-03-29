use axum::Json;
use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use std::collections::HashMap;
use tracing::instrument;

use crate::error::AppError;
use crate::schemas::{
    TopicCoverageQuery, TopicCoverageResponse, TopicGapsQuery, TopicGapsResponse,
    TopicWeakNotesQuery, TopicWeakNotesResponse, TopicsTreeQuery, TopicsTreeResponse,
};
use crate::state::AppState;
use common::error::AnkiAtlasError;
use surface_contracts::analytics::GapKind;

#[instrument(skip(state))]
pub async fn topics(
    State(state): State<AppState>,
    Query(query): Query<TopicsTreeQuery>,
) -> Result<Response, AppError> {
    let topics = state
        .services
        .analytics
        .get_taxonomy_tree(query.root_path)
        .await
        .map_err(AppError::from)?;

    Ok(Json(TopicsTreeResponse { topics }).into_response())
}

#[instrument(skip(state))]
pub async fn topic_coverage(
    State(state): State<AppState>,
    Query(query): Query<TopicCoverageQuery>,
) -> Result<Response, AppError> {
    let coverage = state
        .services
        .analytics
        .get_coverage(query.topic_path.clone(), query.include_subtree)
        .await
        .map_err(AppError::from)?;

    let coverage = coverage.ok_or_else(|| {
        AppError(
            AnkiAtlasError::NotFound {
                message: format!("topic not found: {}", query.topic_path),
                context: HashMap::new(),
            }
            .into(),
        )
    })?;

    Ok(Json(TopicCoverageResponse::from(coverage)).into_response())
}

#[instrument(skip(state))]
pub async fn topic_gaps(
    State(state): State<AppState>,
    Query(query): Query<TopicGapsQuery>,
) -> Result<Response, AppError> {
    query.validate().map_err(AppError::bad_request)?;
    ensure_topic_exists(&state, &query.topic_path).await?;

    let gaps = state
        .services
        .analytics
        .get_gaps(query.topic_path.clone(), query.min_coverage)
        .await
        .map_err(AppError::from)?;

    let missing_count = gaps
        .iter()
        .filter(|gap| matches!(gap.gap_type, GapKind::Missing))
        .count();
    let undercovered_count = gaps.len().saturating_sub(missing_count);

    Ok(Json(TopicGapsResponse {
        root_path: query.topic_path,
        min_coverage: query.min_coverage,
        gaps: gaps.into_iter().map(Into::into).collect(),
        missing_count,
        undercovered_count,
    })
    .into_response())
}

#[instrument(skip(state))]
pub async fn topic_weak_notes(
    State(state): State<AppState>,
    Query(query): Query<TopicWeakNotesQuery>,
) -> Result<Response, AppError> {
    query.validate().map_err(AppError::bad_request)?;
    ensure_topic_exists(&state, &query.topic_path).await?;

    let notes = state
        .services
        .analytics
        .get_weak_notes(query.topic_path.clone(), query.max_results)
        .await
        .map_err(AppError::from)?;

    Ok(Json(TopicWeakNotesResponse {
        topic_path: query.topic_path,
        max_results: query.max_results,
        notes: notes.into_iter().map(Into::into).collect(),
    })
    .into_response())
}

async fn ensure_topic_exists(state: &AppState, topic_path: &str) -> Result<(), AppError> {
    let exists = state
        .services
        .analytics
        .get_coverage(topic_path.to_string(), false)
        .await
        .map_err(AppError::from)?
        .is_some();

    if exists {
        Ok(())
    } else {
        Err(AppError(
            AnkiAtlasError::NotFound {
                message: format!("topic not found: {topic_path}"),
                context: HashMap::new(),
            }
            .into(),
        ))
    }
}

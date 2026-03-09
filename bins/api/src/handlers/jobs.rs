use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use std::collections::HashMap;
use tracing::instrument;

use crate::error::AppError;
use crate::schemas::{AsyncIndexRequest, AsyncSyncRequest, JobAcceptedResponse, JobStatusResponse};
use crate::state::AppState;

/// Convert a `JobRecord` to a `JobAcceptedResponse` for 202 replies.
fn record_to_accepted(rec: &jobs::JobRecord) -> JobAcceptedResponse {
    JobAcceptedResponse {
        job_id: rec.job_id.clone(),
        status: rec.status,
        job_type: rec.job_type,
        created_at: rec.created_at.unwrap_or_else(chrono::Utc::now),
        scheduled_for: rec.scheduled_for,
        poll_url: format!("/jobs/{}", rec.job_id),
    }
}


/// Map `JobError` to `AppError`, translating backend-unavailable to a 503 domain error.
fn map_job_error(e: jobs::JobError) -> AppError {
    match e {
        jobs::JobError::BackendUnavailable(msg) => AppError(
            common::error::AnkiAtlasError::JobBackendUnavailable {
                message: msg,
                context: HashMap::new(),
            }
            .into(),
        ),
        other => AppError(anyhow::anyhow!(other)),
    }
}

/// Build a 404 JSON response for a missing job.
fn job_not_found(job_id: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "error": "NotFound", "message": format!("job {} not found", job_id) })),
    )
        .into_response()
}

/// Enqueue an async sync job. Returns 202 with job details.
#[instrument(skip(state, req))]
pub async fn enqueue_sync_job(
    State(state): State<AppState>,
    Json(req): Json<AsyncSyncRequest>,
) -> Result<Response, AppError> {
    let mut payload = HashMap::new();
    payload.insert("source".to_string(), serde_json::Value::String(req.source));
    payload.insert(
        "run_migrations".to_string(),
        serde_json::Value::Bool(req.run_migrations),
    );
    payload.insert("index".to_string(), serde_json::Value::Bool(req.index));
    payload.insert(
        "force_reindex".to_string(),
        serde_json::Value::Bool(req.force_reindex),
    );

    let rec = state
        .job_manager
        .enqueue_sync_job(payload, req.run_at)
        .await
        .map_err(map_job_error)?;

    Ok((StatusCode::ACCEPTED, Json(record_to_accepted(&rec))).into_response())
}

/// Enqueue an async index job. Returns 202 with job details.
#[instrument(skip(state, req))]
pub async fn enqueue_index_job(
    State(state): State<AppState>,
    Json(req): Json<AsyncIndexRequest>,
) -> Result<Response, AppError> {
    let mut payload = HashMap::new();
    payload.insert(
        "force_reindex".to_string(),
        serde_json::Value::Bool(req.force_reindex),
    );

    let rec = state
        .job_manager
        .enqueue_index_job(payload, req.run_at)
        .await
        .map_err(map_job_error)?;

    Ok((StatusCode::ACCEPTED, Json(record_to_accepted(&rec))).into_response())
}

/// Get the status of a job by ID. Returns 404 if not found.
#[instrument(skip(state))]
pub async fn get_job_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Response, AppError> {
    let rec = state
        .job_manager
        .get_job(&job_id)
        .await
        .map_err(|e| AppError(anyhow::anyhow!(e)))?;

    match rec {
        Some(r) => Ok((StatusCode::OK, Json(JobStatusResponse::from(r))).into_response()),
        None => Ok(job_not_found(&job_id)),
    }
}

/// Request cancellation of a job by ID. Returns 404 if not found.
#[instrument(skip(state))]
pub async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Response, AppError> {
    let rec = state
        .job_manager
        .cancel_job(&job_id)
        .await
        .map_err(|e| AppError(anyhow::anyhow!(e)))?;

    match rec {
        Some(r) => Ok((StatusCode::OK, Json(JobStatusResponse::from(r))).into_response()),
        None => Ok(job_not_found(&job_id)),
    }
}

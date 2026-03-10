use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
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

/// Map `JobError` to `AppError` with one consistent domain-to-HTTP contract.
fn map_job_error(error: jobs::JobError) -> AppError {
    match error {
        jobs::JobError::BackendUnavailable(message) => AppError(
            common::error::AnkiAtlasError::JobBackendUnavailable {
                message,
                context: HashMap::new(),
            }
            .into(),
        ),
        jobs::JobError::NotFound(job_id) => AppError(
            common::error::AnkiAtlasError::NotFound {
                message: format!("job {job_id} not found"),
                context: HashMap::new(),
            }
            .into(),
        ),
        jobs::JobError::TerminalState { job_id, status } => AppError(
            common::error::AnkiAtlasError::Conflict {
                message: format!("job {job_id} already in terminal state: {status}"),
                context: HashMap::new(),
            }
            .into(),
        ),
        jobs::JobError::Unsupported(message) => AppError(
            common::error::AnkiAtlasError::Conflict {
                message,
                context: HashMap::new(),
            }
            .into(),
        ),
        other => AppError(anyhow::anyhow!(other)),
    }
}

/// Enqueue an async sync job. Returns 202 with job details.
#[instrument(skip(state, req))]
pub async fn enqueue_sync_job(
    State(state): State<AppState>,
    Json(req): Json<AsyncSyncRequest>,
) -> Result<Response, AppError> {
    let run_at = req.run_at;
    let payload = jobs::SyncJobPayload::from(req);

    let rec = state
        .services
        .job_manager
        .enqueue_sync_job(payload, run_at)
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
    let run_at = req.run_at;
    let payload = jobs::IndexJobPayload::from(req);

    let rec = state
        .services
        .job_manager
        .enqueue_index_job(payload, run_at)
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
        .services
        .job_manager
        .get_job(&job_id)
        .await
        .map_err(map_job_error)?;

    Ok((StatusCode::OK, Json(JobStatusResponse::from(rec))).into_response())
}

/// Request cancellation of a job by ID. Returns 404 if not found.
#[instrument(skip(state))]
pub async fn cancel_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Response, AppError> {
    let rec = state
        .services
        .job_manager
        .cancel_job(&job_id)
        .await
        .map_err(map_job_error)?;

    Ok((StatusCode::OK, Json(JobStatusResponse::from(rec))).into_response())
}

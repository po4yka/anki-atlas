use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use std::collections::HashMap;

use crate::error::AppError;
use crate::schemas::{
    AsyncIndexRequest, AsyncSyncRequest, JobAcceptedResponse, JobStatusResponse,
};
use crate::state::AppState;

fn record_to_accepted(rec: &jobs::JobRecord) -> JobAcceptedResponse {
    JobAcceptedResponse {
        job_id: rec.job_id.clone(),
        status: rec.status.to_string(),
        job_type: rec.job_type.to_string(),
        created_at: rec.created_at.unwrap_or_else(chrono::Utc::now),
        scheduled_for: rec.scheduled_for,
        poll_url: format!("/jobs/{}", rec.job_id),
    }
}

fn record_to_status(rec: &jobs::JobRecord) -> JobStatusResponse {
    JobStatusResponse {
        job_id: rec.job_id.clone(),
        job_type: rec.job_type.to_string(),
        status: rec.status.to_string(),
        progress: rec.progress,
        message: rec.message.clone(),
        attempts: rec.attempts,
        max_retries: rec.max_retries,
        cancel_requested: rec.cancel_requested,
        created_at: rec.created_at,
        scheduled_for: rec.scheduled_for,
        started_at: rec.started_at,
        finished_at: rec.finished_at,
        result: rec.result.clone(),
        error: rec.error.clone(),
    }
}

pub async fn enqueue_sync_job(
    State(state): State<AppState>,
    Json(req): Json<AsyncSyncRequest>,
) -> Result<Response, AppError> {
    let mut payload = HashMap::new();
    payload.insert(
        "source".to_string(),
        serde_json::Value::String(req.source),
    );
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
        .map_err(|e| match e {
            jobs::JobError::BackendUnavailable(msg) => AppError(
                common::error::AnkiAtlasError::JobBackendUnavailable {
                    message: msg,
                    context: HashMap::new(),
                }
                .into(),
            ),
            other => AppError(anyhow::anyhow!(other)),
        })?;

    Ok((StatusCode::ACCEPTED, Json(record_to_accepted(&rec))).into_response())
}

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
        .map_err(|e| match e {
            jobs::JobError::BackendUnavailable(msg) => AppError(
                common::error::AnkiAtlasError::JobBackendUnavailable {
                    message: msg,
                    context: HashMap::new(),
                }
                .into(),
            ),
            other => AppError(anyhow::anyhow!(other)),
        })?;

    Ok((StatusCode::ACCEPTED, Json(record_to_accepted(&rec))).into_response())
}

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
        Some(r) => Ok((StatusCode::OK, Json(record_to_status(&r))).into_response()),
        None => Ok((StatusCode::NOT_FOUND, Json(json!({ "error": "NotFound", "message": format!("job {} not found", job_id) }))).into_response()),
    }
}

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
        Some(r) => Ok((StatusCode::OK, Json(record_to_status(&r))).into_response()),
        None => Ok((StatusCode::NOT_FOUND, Json(json!({ "error": "NotFound", "message": format!("job {} not found", job_id) }))).into_response()),
    }
}

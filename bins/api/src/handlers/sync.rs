use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;
use tracing::instrument;

use crate::schemas::{IndexRequest, IndexResponse, SyncRequest};

/// Synchronous sync endpoint. Validates the source path and extension.
#[instrument(skip(body))]
pub async fn sync(body: Result<Json<SyncRequest>, JsonRejection>) -> Response {
    let Json(req) = match body {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "BadRequest", "message": e.body_text() })),
            )
                .into_response();
        }
    };

    let path = std::path::Path::new(&req.source);

    // Validate extension
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default();
    if ext != "anki2" && ext != "anki21" {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "InvalidExtension", "message": "source must have .anki2 or .anki21 extension" })),
        )
            .into_response();
    }

    // Validate path exists
    if !path.exists() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "NotFound", "message": "source path does not exist" })),
        )
            .into_response();
    }

    // Stub: real implementation would call sync service
    let _ = req;
    (
        StatusCode::OK,
        Json(json!({ "status": "completed", "message": "sync not yet implemented" })),
    )
        .into_response()
}

/// Index all notes. Returns processing statistics.
#[instrument(skip(_req))]
pub async fn index_notes(Json(_req): Json<IndexRequest>) -> Json<IndexResponse> {
    Json(IndexResponse {
        status: "completed".into(),
        notes_processed: 0,
        notes_embedded: 0,
        notes_skipped: 0,
        notes_deleted: 0,
        errors: vec![],
    })
}

use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

pub async fn list_topics() -> Json<Value> {
    Json(json!({ "topics": [] }))
}

/// Handles /topics/{*rest} and dispatches based on suffix.
/// Routes: /topics/<path>/coverage, /topics/<path>/gaps
pub async fn topic_wildcard(Path(rest): Path<String>) -> Response {
    if let Some(topic_path) = rest.strip_suffix("/coverage") {
        topic_coverage(topic_path).await.into_response()
    } else if let Some(topic_path) = rest.strip_suffix("/gaps") {
        topic_gaps(topic_path).await.into_response()
    } else {
        (StatusCode::NOT_FOUND, Json(json!({ "error": "not found" }))).into_response()
    }
}

async fn topic_coverage(topic_path: &str) -> Json<Value> {
    Json(json!({
        "topic_path": topic_path,
        "status": "not_implemented",
    }))
}

async fn topic_gaps(topic_path: &str) -> Json<Value> {
    Json(json!({
        "topic_path": topic_path,
        "status": "not_implemented",
    }))
}

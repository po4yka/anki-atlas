use axum::Json;
use serde_json::{json, Value};
use tracing::instrument;

/// Returns information about the current search index state.
#[instrument]
pub async fn index_info() -> Json<Value> {
    Json(json!({
        "status": "not_implemented",
    }))
}

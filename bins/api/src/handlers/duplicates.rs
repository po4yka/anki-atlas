use axum::Json;
use serde_json::{Value, json};
use tracing::instrument;

/// Find duplicate note clusters based on similarity threshold.
#[instrument]
pub async fn find_duplicates() -> Json<Value> {
    Json(json!({
        "clusters": [],
        "stats": {},
    }))
}

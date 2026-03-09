use axum::Json;
use serde_json::{json, Value};

pub async fn find_duplicates() -> Json<Value> {
    Json(json!({
        "clusters": [],
        "stats": {},
    }))
}

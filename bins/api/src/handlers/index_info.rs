use axum::Json;
use serde_json::{json, Value};

pub async fn index_info() -> Json<Value> {
    Json(json!({
        "status": "not_implemented",
    }))
}

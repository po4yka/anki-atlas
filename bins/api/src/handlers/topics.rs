use axum::extract::Path;
use axum::response::Response;
use tracing::instrument;

use crate::error::AppError;

/// List all top-level topics.
#[instrument]
pub async fn list_topics() -> Result<Response, AppError> {
    Err(super::unwired_surface(
        "the /topics endpoint",
        "wire it through the analytics taxonomy service before exposing it",
    ))
}

/// Handles `/topics/{*rest}` and dispatches based on suffix.
///
/// Routes: `/topics/<path>/coverage`, `/topics/<path>/gaps`
#[instrument(skip_all)]
pub async fn topic_wildcard(Path(rest): Path<String>) -> Result<Response, AppError> {
    if let Some(topic_path) = rest.strip_suffix("/coverage") {
        topic_coverage(topic_path).await
    } else if let Some(topic_path) = rest.strip_suffix("/gaps") {
        topic_gaps(topic_path).await
    } else {
        Err(super::unwired_surface(
            "the /topics wildcard endpoint",
            "use /topics/<path>/coverage or /topics/<path>/gaps once analytics wiring exists",
        ))
    }
}

async fn topic_coverage(topic_path: &str) -> Result<Response, AppError> {
    Err(super::unwired_surface(
        &format!("the /topics/{topic_path}/coverage endpoint"),
        "wire it through the analytics coverage service before exposing it",
    ))
}

async fn topic_gaps(topic_path: &str) -> Result<Response, AppError> {
    Err(super::unwired_surface(
        &format!("the /topics/{topic_path}/gaps endpoint"),
        "wire it through the analytics gaps service before exposing it",
    ))
}

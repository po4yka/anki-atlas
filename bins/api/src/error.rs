use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use common::error::AnkiAtlasError;
use serde_json::json;

/// Wrapper that maps domain errors to HTTP responses.
pub struct AppError(pub anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = if let Some(e) = self.0.downcast_ref::<AnkiAtlasError>()
        {
            match e {
                AnkiAtlasError::NotFound { message, .. } => {
                    (StatusCode::NOT_FOUND, "NotFound", message.clone())
                }
                AnkiAtlasError::Conflict { message, .. } => {
                    (StatusCode::CONFLICT, "Conflict", message.clone())
                }
                AnkiAtlasError::DatabaseConnection { message, .. } => {
                    (StatusCode::SERVICE_UNAVAILABLE, "DatabaseConnection", message.clone())
                }
                AnkiAtlasError::VectorStoreConnection { message, .. } => {
                    (StatusCode::SERVICE_UNAVAILABLE, "VectorStoreConnection", message.clone())
                }
                AnkiAtlasError::JobBackendUnavailable { message, .. } => {
                    (StatusCode::SERVICE_UNAVAILABLE, "JobBackendUnavailable", message.clone())
                }
                other => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "InternalError",
                    other.to_string(),
                ),
            }
        } else {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "InternalError",
                self.0.to_string(),
            )
        };

        let body = json!({
            "error": error_type,
            "message": message,
        });

        (status, axum::Json(body)).into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

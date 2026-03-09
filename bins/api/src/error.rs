use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

/// Wrapper that maps domain errors to HTTP responses.
pub struct AppError(pub anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        // TODO(impl): map domain errors to correct status codes
        (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

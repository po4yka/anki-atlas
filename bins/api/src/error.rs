use analytics::AnalyticsError;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use common::error::AnkiAtlasError;
use search::error::{RerankError, SearchError};
use serde_json::{Value, json};
use std::collections::HashMap;

/// Wrapper that maps domain and validation errors to HTTP responses.
pub struct AppError(pub anyhow::Error);

impl AppError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self(
            AnkiAtlasError::Configuration {
                message: message.into(),
                context: HashMap::new(),
            }
            .into(),
        )
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type, message, details) =
            if let Some(e) = self.0.downcast_ref::<AnkiAtlasError>() {
                match e {
                    AnkiAtlasError::NotFound { message, context } => (
                        StatusCode::NOT_FOUND,
                        "NotFound",
                        message.clone(),
                        context_to_details(context),
                    ),
                    AnkiAtlasError::Conflict { message, context } => (
                        StatusCode::CONFLICT,
                        "Conflict",
                        message.clone(),
                        context_to_details(context),
                    ),
                    AnkiAtlasError::Configuration { message, context } => (
                        StatusCode::BAD_REQUEST,
                        "BadRequest",
                        message.clone(),
                        context_to_details(context),
                    ),
                    AnkiAtlasError::DatabaseConnection { message, context } => (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "DatabaseConnection",
                        message.clone(),
                        context_to_details(context),
                    ),
                    AnkiAtlasError::VectorStoreConnection { message, context } => (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "VectorStoreConnection",
                        message.clone(),
                        context_to_details(context),
                    ),
                    AnkiAtlasError::JobBackendUnavailable { message, context } => (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "JobBackendUnavailable",
                        message.clone(),
                        context_to_details(context),
                    ),
                    other => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "InternalError",
                        other.to_string(),
                        None,
                    ),
                }
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "InternalError",
                    self.0.to_string(),
                    None,
                )
            };

        let mut body = serde_json::Map::new();
        body.insert("error".to_string(), Value::String(error_type.to_string()));
        body.insert("message".to_string(), Value::String(message));
        if let Some(details) = details {
            body.insert("details".to_string(), details);
        }

        (status, axum::Json(Value::Object(body))).into_response()
    }
}

impl From<SearchError> for AppError {
    fn from(error: SearchError) -> Self {
        match error {
            SearchError::Database(source) => Self(
                AnkiAtlasError::DatabaseConnection {
                    message: source.to_string(),
                    context: HashMap::new(),
                }
                .into(),
            ),
            SearchError::Embedding(source) => Self(anyhow::Error::new(source)),
            SearchError::VectorStore(source) => Self(
                AnkiAtlasError::VectorStoreConnection {
                    message: source.to_string(),
                    context: HashMap::new(),
                }
                .into(),
            ),
            SearchError::Rerank(RerankError::Transport { message })
            | SearchError::Rerank(RerankError::Http { body: message, .. })
            | SearchError::Rerank(RerankError::Protocol { message }) => Self(
                AnkiAtlasError::Provider {
                    message,
                    context: HashMap::new(),
                }
                .into(),
            ),
        }
    }
}

impl From<AnalyticsError> for AppError {
    fn from(error: AnalyticsError) -> Self {
        match error {
            AnalyticsError::Database(source) => Self(
                AnkiAtlasError::DatabaseConnection {
                    message: source.to_string(),
                    context: HashMap::new(),
                }
                .into(),
            ),
            AnalyticsError::Embedding(source) => Self(anyhow::Error::new(source)),
            AnalyticsError::VectorStore(source) => Self(
                AnkiAtlasError::VectorStoreConnection {
                    message: source.to_string(),
                    context: HashMap::new(),
                }
                .into(),
            ),
            AnalyticsError::TopicNotFound(topic_path) => Self(
                AnkiAtlasError::NotFound {
                    message: format!("topic not found: {topic_path}"),
                    context: HashMap::new(),
                }
                .into(),
            ),
            other => Self(anyhow::Error::new(other)),
        }
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        Self(err)
    }
}

impl From<AnkiAtlasError> for AppError {
    fn from(err: AnkiAtlasError) -> Self {
        Self(err.into())
    }
}

fn context_to_details(context: &HashMap<String, String>) -> Option<Value> {
    if context.is_empty() {
        None
    } else {
        Some(json!(context))
    }
}

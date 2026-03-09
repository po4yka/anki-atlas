use anki_atlas_api::error::AppError;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use common::error::AnkiAtlasError;
use std::collections::HashMap;

fn context() -> HashMap<String, String> {
    HashMap::new()
}

#[tokio::test]
async fn not_found_maps_to_404() {
    let err = AnkiAtlasError::NotFound {
        message: "topic not found".into(),
        context: context(),
    };
    let response = AppError(err.into()).into_response();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn conflict_maps_to_409() {
    let err = AnkiAtlasError::Conflict {
        message: "already exists".into(),
        context: context(),
    };
    let response = AppError(err.into()).into_response();
    assert_eq!(response.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn database_error_maps_to_503() {
    let err = AnkiAtlasError::DatabaseConnection {
        message: "connection refused".into(),
        context: context(),
    };
    let response = AppError(err.into()).into_response();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn vector_store_error_maps_to_503() {
    let err = AnkiAtlasError::VectorStoreConnection {
        message: "qdrant down".into(),
        context: context(),
    };
    let response = AppError(err.into()).into_response();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn job_backend_unavailable_maps_to_503() {
    let err = AnkiAtlasError::JobBackendUnavailable {
        message: "redis down".into(),
        context: context(),
    };
    let response = AppError(err.into()).into_response();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn other_error_maps_to_500() {
    let err = AnkiAtlasError::Sync {
        message: "unexpected".into(),
        context: context(),
    };
    let response = AppError(err.into()).into_response();
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn error_response_has_json_body() {
    let err = AnkiAtlasError::NotFound {
        message: "thing not found".into(),
        context: context(),
    };
    let response = AppError(err.into()).into_response();
    let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(v["error"].is_string(), "response should have 'error' field");
    assert!(
        v["message"].is_string(),
        "response should have 'message' field"
    );
}

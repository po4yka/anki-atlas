use anki_atlas_api::middleware::{ApiKeyLayer, CorrelationIdLayer};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use tower::ServiceExt;

async fn ok_handler() -> impl IntoResponse {
    "ok"
}

fn test_router_with_correlation_id() -> Router {
    Router::new()
        .route("/test", get(ok_handler))
        .layer(CorrelationIdLayer)
}

fn test_router_with_api_key(key: Option<String>) -> Router {
    Router::new()
        .route("/test", get(ok_handler))
        .layer(ApiKeyLayer::new(key))
}

// --- Correlation ID ---

#[tokio::test]
async fn correlation_id_generated_when_not_provided() {
    let app = test_router_with_correlation_id();
    let req = Request::builder()
        .uri("/test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(
        resp.headers().contains_key("x-request-id"),
        "response should contain X-Request-ID header"
    );
}

#[tokio::test]
async fn correlation_id_preserved_from_request() {
    let app = test_router_with_correlation_id();
    let req = Request::builder()
        .uri("/test")
        .header("x-request-id", "my-custom-id-123")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.headers().get("x-request-id").unwrap(),
        "my-custom-id-123"
    );
}

// --- API Key ---

#[tokio::test]
async fn api_key_passes_when_not_configured() {
    let app = test_router_with_api_key(None);
    let req = Request::builder()
        .uri("/test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn api_key_blocks_when_missing() {
    let app = test_router_with_api_key(Some("secret-key".into()));
    let req = Request::builder()
        .uri("/test")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn api_key_blocks_wrong_key() {
    let app = test_router_with_api_key(Some("secret-key".into()));
    let req = Request::builder()
        .uri("/test")
        .header("x-api-key", "wrong-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn api_key_passes_correct_key() {
    let app = test_router_with_api_key(Some("secret-key".into()));
    let req = Request::builder()
        .uri("/test")
        .header("x-api-key", "secret-key")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

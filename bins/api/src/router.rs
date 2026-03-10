use axum::Router;
use axum::routing::{get, post};

use crate::handlers;
use crate::middleware::{ApiKeyLayer, CorrelationIdLayer};
use crate::state::AppState;

/// Build the application router with all routes.
///
/// Public routes (health, ready) are not behind API key auth.
/// All other routes require a valid API key when configured.
pub fn build_router(state: AppState) -> Router {
    let api_key = state.api.api_key.clone();

    // Public routes - no auth required
    let public = Router::new()
        .route("/health", get(handlers::health))
        .route("/ready", get(handlers::ready));

    // Protected routes - require API key when configured
    let protected = Router::new()
        // Sync operations
        .route("/sync", post(handlers::sync))
        .route("/index", post(handlers::index_notes))
        // Async jobs
        .route("/jobs/sync", post(handlers::enqueue_sync_job))
        .route("/jobs/index", post(handlers::enqueue_index_job))
        .route("/jobs/{job_id}", get(handlers::get_job_status))
        .route("/jobs/{job_id}/cancel", post(handlers::cancel_job))
        // Search
        .route("/search", post(handlers::search))
        // Topics
        .route("/topics", get(handlers::list_topics))
        .route("/topics/{*rest}", get(handlers::topic_wildcard))
        // Duplicates
        .route("/duplicates", get(handlers::find_duplicates))
        // Index info
        .route("/index/info", get(handlers::index_info))
        .layer(ApiKeyLayer::new(api_key));

    public
        .merge(protected)
        .layer(CorrelationIdLayer)
        .with_state(state)
}

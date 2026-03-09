use axum::Router;
use crate::state::AppState;

/// Build the application router with all routes.
pub fn build_router(_state: AppState) -> Router {
    // TODO(impl): add all routes with handlers
    Router::new()
}

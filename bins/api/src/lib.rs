pub mod error;
pub mod handlers;
pub mod middleware;
pub mod router;
pub mod schemas;
pub mod services;
pub mod state;

#[cfg(test)]
mod send_sync_tests {
    common::assert_send_sync!(
        super::error::AppError,
        super::state::AppState,
        super::services::ApiServices,
        super::middleware::CorrelationIdLayer,
        super::middleware::ApiKeyLayer,
        super::schemas::SearchRequest,
        super::schemas::SearchResponse,
        super::schemas::JobAcceptedResponse,
        super::schemas::JobStatusResponse,
        super::schemas::DuplicatesResponse,
    );
}

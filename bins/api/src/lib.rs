pub mod error;
pub mod handlers;
pub mod middleware;
pub mod router;
pub mod schemas;
pub mod services;
pub mod state;

#[cfg(test)]
mod send_sync_tests {
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn public_types_are_send_and_sync() {
        assert_send_sync::<super::error::AppError>();
        assert_send_sync::<super::state::AppState>();
        assert_send_sync::<super::services::ApiServices>();
        assert_send_sync::<super::middleware::CorrelationIdLayer>();
        assert_send_sync::<super::middleware::ApiKeyLayer>();
        assert_send_sync::<super::schemas::SearchRequest>();
        assert_send_sync::<super::schemas::SearchResponse>();
        assert_send_sync::<super::schemas::JobAcceptedResponse>();
        assert_send_sync::<super::schemas::JobStatusResponse>();
        assert_send_sync::<super::schemas::DuplicatesResponse>();
    }
}

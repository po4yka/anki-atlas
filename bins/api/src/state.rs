use crate::services::ApiServices;
use common::config::ApiSettings;
use std::sync::Arc;

/// Shared application state accessible in all handlers.
#[derive(Clone)]
pub struct AppState {
    pub api: Arc<ApiSettings>,
    pub services: Arc<ApiServices>,
}

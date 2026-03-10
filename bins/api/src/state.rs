use common::config::ApiSettings;
use jobs::JobManager;
use std::sync::Arc;

/// Shared application state accessible in all handlers.
#[derive(Clone)]
pub struct AppState {
    pub api: Arc<ApiSettings>,
    pub job_manager: Arc<dyn JobManager>,
}

use common::config::Settings;
use jobs::JobManager;
use std::sync::Arc;

/// Shared application state accessible in all handlers.
#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<Settings>,
    pub job_manager: Arc<dyn JobManager>,
}

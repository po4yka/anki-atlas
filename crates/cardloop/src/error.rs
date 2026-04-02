/// Errors produced by the cardloop crate.
#[derive(Debug, thiserror::Error)]
pub enum CardloopError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("no active session")]
    NoSession,

    #[error("work item not found: {0}")]
    NotFound(String),

    #[error("invalid transition {from} -> {to} for {id}")]
    InvalidTransition {
        id: String,
        from: String,
        to: String,
    },

    #[error("registry error: {0}")]
    Registry(#[from] card::registry::RegistryError),

    #[error("validation error: {0}")]
    Validation(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

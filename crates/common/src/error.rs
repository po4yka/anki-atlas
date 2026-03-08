use std::collections::HashMap;
use thiserror::Error;

/// Optional context attached to any error variant.
pub type ErrorContext = HashMap<String, String>;

#[derive(Debug, Error)]
pub enum AnkiAtlasError {
    #[error("TODO")]
    DatabaseConnection { message: String, context: ErrorContext },

    #[error("TODO")]
    Migration { message: String, context: ErrorContext },

    #[error("TODO")]
    VectorStoreConnection { message: String, context: ErrorContext },

    #[error("TODO")]
    Collection { message: String, context: ErrorContext },

    #[error("TODO")]
    DimensionMismatch {
        collection: String,
        expected: u32,
        actual: u32,
    },

    #[error("TODO")]
    Embedding { message: String, context: ErrorContext },

    #[error("TODO")]
    EmbeddingApi { message: String, context: ErrorContext },

    #[error("TODO")]
    EmbeddingTimeout { message: String, context: ErrorContext },

    #[error("TODO")]
    EmbeddingModelChanged { stored: String, current: String },

    #[error("TODO")]
    Sync { message: String, context: ErrorContext },

    #[error("TODO")]
    CollectionNotFound { message: String, context: ErrorContext },

    #[error("TODO")]
    SyncConflict { message: String, context: ErrorContext },

    #[error("TODO")]
    AnkiConnect { message: String, context: ErrorContext },

    #[error("TODO")]
    AnkiReader { message: String, context: ErrorContext },

    #[error("TODO")]
    Configuration { message: String, context: ErrorContext },

    #[error("TODO")]
    NotFound { message: String, context: ErrorContext },

    #[error("TODO")]
    Conflict { message: String, context: ErrorContext },

    #[error("TODO")]
    CardGeneration { message: String, context: ErrorContext },

    #[error("TODO")]
    CardValidation { message: String, context: ErrorContext },

    #[error("TODO")]
    Provider { message: String, context: ErrorContext },

    #[error("TODO")]
    ObsidianParse { message: String, context: ErrorContext },

    #[error("TODO")]
    JobBackendUnavailable { message: String, context: ErrorContext },
}

/// Alias used throughout the codebase.
pub type Result<T> = std::result::Result<T, AnkiAtlasError>;

/// Convenience trait for adding context to errors.
pub trait WithContext {
    fn with_context(self, key: impl Into<String>, value: impl Into<String>) -> Self;
}

impl WithContext for AnkiAtlasError {
    fn with_context(self, _key: impl Into<String>, _value: impl Into<String>) -> Self {
        // TODO(ralph): implement context insertion
        self
    }
}

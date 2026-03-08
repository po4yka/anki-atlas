use std::collections::HashMap;
use thiserror::Error;

/// Optional context attached to any error variant.
pub type ErrorContext = HashMap<String, String>;

#[derive(Debug, Error)]
pub enum AnkiAtlasError {
    #[error("database connection failed: {message}")]
    DatabaseConnection { message: String, context: ErrorContext },

    #[error("migration failed: {message}")]
    Migration { message: String, context: ErrorContext },

    #[error("vector store connection failed: {message}")]
    VectorStoreConnection { message: String, context: ErrorContext },

    #[error("collection operation failed: {message}")]
    Collection { message: String, context: ErrorContext },

    #[error("dimension mismatch on '{collection}': expected {expected}, got {actual}")]
    DimensionMismatch {
        collection: String,
        expected: u32,
        actual: u32,
    },

    #[error("embedding error: {message}")]
    Embedding { message: String, context: ErrorContext },

    #[error("embedding API error: {message}")]
    EmbeddingApi { message: String, context: ErrorContext },

    #[error("embedding timeout: {message}")]
    EmbeddingTimeout { message: String, context: ErrorContext },

    #[error("embedding model changed: '{stored}' -> '{current}'. Use --force-reindex.")]
    EmbeddingModelChanged { stored: String, current: String },

    #[error("sync error: {message}")]
    Sync { message: String, context: ErrorContext },

    #[error("collection not found: {message}")]
    CollectionNotFound { message: String, context: ErrorContext },

    #[error("sync conflict: {message}")]
    SyncConflict { message: String, context: ErrorContext },

    #[error("AnkiConnect error: {message}")]
    AnkiConnect { message: String, context: ErrorContext },

    #[error("Anki reader error: {message}")]
    AnkiReader { message: String, context: ErrorContext },

    #[error("configuration error: {message}")]
    Configuration { message: String, context: ErrorContext },

    #[error("not found: {message}")]
    NotFound { message: String, context: ErrorContext },

    #[error("conflict: {message}")]
    Conflict { message: String, context: ErrorContext },

    #[error("card generation error: {message}")]
    CardGeneration { message: String, context: ErrorContext },

    #[error("card validation error: {message}")]
    CardValidation { message: String, context: ErrorContext },

    #[error("provider error: {message}")]
    Provider { message: String, context: ErrorContext },

    #[error("obsidian parse error: {message}")]
    ObsidianParse { message: String, context: ErrorContext },

    #[error("job backend unavailable: {message}")]
    JobBackendUnavailable { message: String, context: ErrorContext },
}

/// Alias used throughout the codebase.
pub type Result<T> = std::result::Result<T, AnkiAtlasError>;

/// Convenience trait for adding context to errors.
pub trait WithContext {
    fn with_context(self, key: impl Into<String>, value: impl Into<String>) -> Self;
}

impl WithContext for AnkiAtlasError {
    fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        macro_rules! insert_context {
            ($ctx:expr) => {{
                $ctx.insert(key.into(), value.into());
            }};
        }
        match &mut self {
            Self::DatabaseConnection { context, .. }
            | Self::Migration { context, .. }
            | Self::VectorStoreConnection { context, .. }
            | Self::Collection { context, .. }
            | Self::Embedding { context, .. }
            | Self::EmbeddingApi { context, .. }
            | Self::EmbeddingTimeout { context, .. }
            | Self::Sync { context, .. }
            | Self::CollectionNotFound { context, .. }
            | Self::SyncConflict { context, .. }
            | Self::AnkiConnect { context, .. }
            | Self::AnkiReader { context, .. }
            | Self::Configuration { context, .. }
            | Self::NotFound { context, .. }
            | Self::Conflict { context, .. }
            | Self::CardGeneration { context, .. }
            | Self::CardValidation { context, .. }
            | Self::Provider { context, .. }
            | Self::ObsidianParse { context, .. }
            | Self::JobBackendUnavailable { context, .. } => {
                insert_context!(context);
            }
            // Variants without context field — noop
            Self::DimensionMismatch { .. } | Self::EmbeddingModelChanged { .. } => {}
        }
        self
    }
}

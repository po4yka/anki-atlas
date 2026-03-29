use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SurfaceError {
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("path not found: {0}")]
    PathNotFound(PathBuf),
    #[error("resource not found: {0}")]
    NotFound(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("job backend error: {0}")]
    Job(#[from] jobs::JobError),
    #[error("sync error: {0}")]
    Sync(#[from] common::error::AnkiAtlasError),
    #[error("index error: {0}")]
    Index(#[from] indexer::service::IndexError),
    #[error("embedding error: {0}")]
    Embedding(#[from] indexer::embeddings::EmbeddingError),
    #[error("vector store error: {0}")]
    VectorStore(#[from] indexer::qdrant::VectorStoreError),
    #[error("provider error: {0}")]
    Provider(String),
    #[error("search error: {0}")]
    Search(#[from] search::error::SearchError),
    #[error("analytics error: {0}")]
    Analytics(#[from] analytics::AnalyticsError),
    #[error("obsidian error: {0}")]
    Obsidian(#[from] obsidian::ObsidianError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_error_display_messages() {
        let err = SurfaceError::Unsupported("nope".into());
        assert_eq!(err.to_string(), "unsupported operation: nope");

        let err = SurfaceError::PathNotFound(PathBuf::from("/tmp/missing"));
        assert_eq!(err.to_string(), "path not found: /tmp/missing");

        let err = SurfaceError::InvalidInput("bad data".into());
        assert_eq!(err.to_string(), "invalid input: bad data");
    }

    #[test]
    fn surface_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let err: SurfaceError = io_err.into();
        assert!(matches!(err, SurfaceError::Io(_)));
        assert!(err.to_string().contains("gone"));
    }

    #[test]
    fn surface_error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SurfaceError>();
    }
}

use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SurfaceError {
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("path not found: {0}")]
    PathNotFound(PathBuf),
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
    #[error("search error: {0}")]
    Search(#[from] search::error::SearchError),
    #[error("analytics error: {0}")]
    Analytics(#[from] analytics::AnalyticsError),
    #[error("obsidian error: {0}")]
    Obsidian(#[from] obsidian::ObsidianError),
}

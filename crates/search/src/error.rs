use thiserror::Error;

/// Search errors.
#[derive(Debug, Error)]
pub enum SearchError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("embedding error: {0}")]
    Embedding(#[from] indexer::embeddings::EmbeddingError),
    #[error("vector store error: {0}")]
    VectorStore(#[from] indexer::qdrant::VectorStoreError),
    #[error("rerank failed: {0}")]
    Rerank(String),
}

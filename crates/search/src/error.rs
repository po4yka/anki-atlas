use thiserror::Error;

#[derive(Debug, Error)]
pub enum RerankError {
    #[error("transport error: {message}")]
    Transport { message: String },
    #[error("http {status}: {body}")]
    Http { status: u16, body: String },
    #[error("protocol error: {message}")]
    Protocol { message: String },
}

/// Search errors.
#[derive(Debug, Error)]
pub enum SearchError {
    #[error("invalid search request: {0}")]
    InvalidRequest(String),
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("embedding error: {0}")]
    Embedding(#[from] indexer::embeddings::EmbeddingError),
    #[error("vector store error: {0}")]
    VectorStore(#[from] indexer::qdrant::VectorStoreError),
    #[error("rerank failed: {0}")]
    Rerank(#[from] RerankError),
}

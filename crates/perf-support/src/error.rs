use thiserror::Error;

#[derive(Debug, Error)]
pub enum PerfError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("migration error: {0}")]
    Migration(#[from] common::error::AnkiAtlasError),

    #[error("vector store error: {0}")]
    VectorStore(#[from] indexer::qdrant::VectorStoreError),

    #[error("embedding error: {0}")]
    Embedding(#[from] indexer::embeddings::EmbeddingError),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("invalid input: {0}")]
    InvalidInput(String),
}

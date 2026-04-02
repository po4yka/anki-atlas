/// Analytics error enum.
#[derive(Debug, thiserror::Error)]
pub enum AnalyticsError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("embedding error: {0}")]
    Embedding(#[from] indexer::embeddings::EmbeddingError),
    #[error("vector store error: {0}")]
    VectorStore(#[from] indexer::qdrant::VectorStoreError),
    #[error("yaml parse error: {0}")]
    YamlParse(#[from] serde_yml::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("topic not found: {0}")]
    TopicNotFound(String),
}

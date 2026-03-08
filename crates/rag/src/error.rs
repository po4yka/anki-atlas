use thiserror::Error;

#[derive(Debug, Error)]
pub enum RagError {
    #[error("vector store error: {0}")]
    Store(String),

    #[error("chunking error: {0}")]
    Chunking(String),

    #[error("search error: {0}")]
    Search(String),
}

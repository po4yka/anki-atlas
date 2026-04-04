use thiserror::Error;

#[derive(Debug, Error)]
pub enum KnowledgeGraphError {
    #[error("database error: {0}")]
    Database(String),

    #[error("discovery error: {0}")]
    Discovery(String),

    #[error("query error: {0}")]
    Query(String),
}

impl From<sqlx::Error> for KnowledgeGraphError {
    fn from(e: sqlx::Error) -> Self {
        Self::Database(e.to_string())
    }
}

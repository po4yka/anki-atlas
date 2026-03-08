use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::RagError;

/// Result from vector similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub content: String,
    pub score: f32,
    pub source_file: String,
    pub metadata: HashMap<String, String>,
}

impl SearchResult {
    /// Convert distance to similarity (0-1): 1 / (1 + distance).
    pub fn similarity(&self) -> f32 {
        1.0 / (1.0 + self.score)
    }
}

/// Filter clause for metadata queries.
#[derive(Debug, Clone, Default)]
pub struct MetadataFilter {
    pub field: String,
    pub value: String,
}

/// Store-level statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStats {
    pub total_chunks: usize,
    pub unique_files: usize,
    pub topics: Vec<String>,
}

/// Abstract vector store trait for DI.
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait VectorStore: Send + Sync {
    /// Add documents with embeddings. Returns count of newly added.
    async fn add(
        &self,
        ids: &[String],
        documents: &[String],
        embeddings: &[Vec<f32>],
        metadatas: Option<&[HashMap<String, String>]>,
    ) -> Result<usize, RagError>;

    /// Delete all chunks from a specific source file.
    async fn delete_by_source(&self, source_file: &str) -> Result<usize, RagError>;

    /// Reset (drop and recreate) the collection.
    async fn reset(&self) -> Result<(), RagError>;

    /// Search by embedding vector.
    async fn search(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
        min_similarity: f32,
    ) -> Result<Vec<SearchResult>, RagError>;

    /// Return total document count.
    async fn count(&self) -> Result<usize, RagError>;

    /// Return store statistics.
    async fn get_stats(&self) -> Result<StoreStats, RagError>;
}

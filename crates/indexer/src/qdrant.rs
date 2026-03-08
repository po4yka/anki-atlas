use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Payload stored with each Qdrant point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NotePayload {
    pub note_id: i64,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
    pub model_id: i64,
    pub content_hash: String,
    #[serde(default)]
    pub mature: bool,
    #[serde(default)]
    pub lapses: i32,
    #[serde(default)]
    pub reps: i32,
    #[serde(default)]
    pub fail_rate: Option<f64>,
}

/// Sparse vector (indices + values) for BM25-style retrieval.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// Errors from Qdrant operations.
#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("dimension mismatch: collection {collection} expects {expected}, got {actual}")]
    DimensionMismatch {
        collection: String,
        expected: usize,
        actual: usize,
    },
    #[error("qdrant error: {0}")]
    Client(String),
    #[error("connection failed: {0}")]
    Connection(String),
}

/// Result of an upsert batch.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UpsertResult {
    pub upserted: usize,
    pub skipped: usize,
}

/// Search filters passed to Qdrant.
#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    pub deck_names: Option<Vec<String>>,
    pub deck_names_exclude: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub tags_exclude: Option<Vec<String>>,
    pub model_ids: Option<Vec<i64>>,
    pub mature_only: bool,
    pub max_lapses: Option<i32>,
    pub min_reps: Option<i32>,
}

/// Trait for vector store operations. Enables mocking in tests.
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait VectorRepository: Send + Sync {
    /// Ensure collection exists with the given dense vector dimension.
    /// Returns true if newly created, false if already existed.
    async fn ensure_collection(&self, dimension: usize) -> Result<bool, VectorStoreError>;

    /// Upsert dense vectors + payloads. Optional sparse vectors.
    async fn upsert_vectors(
        &self,
        vectors: &[Vec<f32>],
        payloads: &[NotePayload],
        sparse_vectors: Option<&[SparseVector]>,
    ) -> Result<usize, VectorStoreError>;

    /// Delete points by note IDs.
    async fn delete_vectors(&self, note_ids: &[i64]) -> Result<usize, VectorStoreError>;

    /// Get content hashes for existing note IDs. Returns note_id -> hash.
    async fn get_existing_hashes(
        &self,
        note_ids: &[i64],
    ) -> Result<HashMap<i64, String>, VectorStoreError>;

    /// Semantic search. Returns (note_id, score) pairs.
    async fn search(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<(i64, f32)>, VectorStoreError>;

    /// Find notes similar to a given note.
    async fn find_similar_to_note(
        &self,
        note_id: i64,
        limit: usize,
        min_score: f32,
        deck_names: Option<&[String]>,
        tags: Option<&[String]>,
    ) -> Result<Vec<(i64, f32)>, VectorStoreError>;

    /// Close connection / cleanup.
    async fn close(&self) -> Result<(), VectorStoreError>;
}

/// Concrete Qdrant implementation.
pub struct QdrantRepository {
    _url: String,
    _collection_name: String,
}

impl QdrantRepository {
    /// Connect to Qdrant and create repository.
    pub async fn new(_url: &str, _collection_name: &str) -> Result<Self, VectorStoreError> {
        todo!("QdrantRepository::new not implemented")
    }

    /// Convert text into a hashed sparse vector (blake2b tokens, L2-normalized TF weights).
    pub fn text_to_sparse_vector(_text: &str) -> SparseVector {
        todo!("QdrantRepository::text_to_sparse_vector not implemented")
    }
}

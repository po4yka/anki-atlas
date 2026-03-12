use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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
    #[serde(default = "default_chunk_id")]
    pub chunk_id: String,
    #[serde(default = "default_chunk_kind")]
    pub chunk_kind: String,
    #[serde(default = "default_modality")]
    pub modality: String,
    #[serde(default)]
    pub source_field: Option<String>,
    #[serde(default)]
    pub asset_rel_path: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub preview_label: Option<String>,
}

fn default_chunk_id() -> String {
    "legacy:text_primary".to_string()
}

fn default_chunk_kind() -> String {
    "text_primary".to_string()
}

fn default_modality() -> String {
    "text".to_string()
}

/// Semantic hit returned from chunk-aware vector search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticSearchHit {
    pub note_id: i64,
    pub chunk_id: String,
    pub chunk_kind: String,
    pub modality: String,
    #[serde(default)]
    pub source_field: Option<String>,
    #[serde(default)]
    pub asset_rel_path: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub preview_label: Option<String>,
    pub score: f32,
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
    #[error("reindex required: {reason}")]
    ReindexRequired { reason: String },
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

    /// Return the current collection dimension, or `None` if the collection does not exist.
    async fn collection_dimension(&self) -> Result<Option<usize>, VectorStoreError>;

    /// Drop and recreate the collection with the requested dimension.
    async fn recreate_collection(&self, dimension: usize) -> Result<(), VectorStoreError>;

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

    /// Semantic search against chunk vectors.
    async fn search_chunks(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SemanticSearchHit>, VectorStoreError>;

    /// Semantic search. Returns (note_id, score) pairs.
    async fn search(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<(i64, f32)>, VectorStoreError> {
        let mut by_note = HashMap::<i64, f32>::new();
        for hit in self
            .search_chunks(
                query_vector,
                query_sparse,
                limit.saturating_mul(4).max(limit),
                filters,
            )
            .await?
        {
            by_note
                .entry(hit.note_id)
                .and_modify(|score| {
                    if hit.score > *score {
                        *score = hit.score;
                    }
                })
                .or_insert(hit.score);
        }
        let mut results: Vec<_> = by_note.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        Ok(results)
    }

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

#[async_trait]
impl<T> VectorRepository for &T
where
    T: VectorRepository + ?Sized,
{
    async fn ensure_collection(&self, dimension: usize) -> Result<bool, VectorStoreError> {
        (*self).ensure_collection(dimension).await
    }

    async fn collection_dimension(&self) -> Result<Option<usize>, VectorStoreError> {
        (*self).collection_dimension().await
    }

    async fn recreate_collection(&self, dimension: usize) -> Result<(), VectorStoreError> {
        (*self).recreate_collection(dimension).await
    }

    async fn upsert_vectors(
        &self,
        vectors: &[Vec<f32>],
        payloads: &[NotePayload],
        sparse_vectors: Option<&[SparseVector]>,
    ) -> Result<usize, VectorStoreError> {
        (*self)
            .upsert_vectors(vectors, payloads, sparse_vectors)
            .await
    }

    async fn delete_vectors(&self, note_ids: &[i64]) -> Result<usize, VectorStoreError> {
        (*self).delete_vectors(note_ids).await
    }

    async fn get_existing_hashes(
        &self,
        note_ids: &[i64],
    ) -> Result<HashMap<i64, String>, VectorStoreError> {
        (*self).get_existing_hashes(note_ids).await
    }

    async fn search_chunks(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SemanticSearchHit>, VectorStoreError> {
        (*self)
            .search_chunks(query_vector, query_sparse, limit, filters)
            .await
    }

    async fn search(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<(i64, f32)>, VectorStoreError> {
        (*self)
            .search(query_vector, query_sparse, limit, filters)
            .await
    }

    async fn find_similar_to_note(
        &self,
        note_id: i64,
        limit: usize,
        min_score: f32,
        deck_names: Option<&[String]>,
        tags: Option<&[String]>,
    ) -> Result<Vec<(i64, f32)>, VectorStoreError> {
        (*self)
            .find_similar_to_note(note_id, limit, min_score, deck_names, tags)
            .await
    }

    async fn close(&self) -> Result<(), VectorStoreError> {
        (*self).close().await
    }
}

#[async_trait]
impl<T> VectorRepository for Box<T>
where
    T: VectorRepository + ?Sized,
{
    async fn ensure_collection(&self, dimension: usize) -> Result<bool, VectorStoreError> {
        (**self).ensure_collection(dimension).await
    }

    async fn collection_dimension(&self) -> Result<Option<usize>, VectorStoreError> {
        (**self).collection_dimension().await
    }

    async fn recreate_collection(&self, dimension: usize) -> Result<(), VectorStoreError> {
        (**self).recreate_collection(dimension).await
    }

    async fn upsert_vectors(
        &self,
        vectors: &[Vec<f32>],
        payloads: &[NotePayload],
        sparse_vectors: Option<&[SparseVector]>,
    ) -> Result<usize, VectorStoreError> {
        (**self)
            .upsert_vectors(vectors, payloads, sparse_vectors)
            .await
    }

    async fn delete_vectors(&self, note_ids: &[i64]) -> Result<usize, VectorStoreError> {
        (**self).delete_vectors(note_ids).await
    }

    async fn get_existing_hashes(
        &self,
        note_ids: &[i64],
    ) -> Result<HashMap<i64, String>, VectorStoreError> {
        (**self).get_existing_hashes(note_ids).await
    }

    async fn search_chunks(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SemanticSearchHit>, VectorStoreError> {
        (**self)
            .search_chunks(query_vector, query_sparse, limit, filters)
            .await
    }

    async fn search(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<(i64, f32)>, VectorStoreError> {
        (**self)
            .search(query_vector, query_sparse, limit, filters)
            .await
    }

    async fn find_similar_to_note(
        &self,
        note_id: i64,
        limit: usize,
        min_score: f32,
        deck_names: Option<&[String]>,
        tags: Option<&[String]>,
    ) -> Result<Vec<(i64, f32)>, VectorStoreError> {
        (**self)
            .find_similar_to_note(note_id, limit, min_score, deck_names, tags)
            .await
    }

    async fn close(&self) -> Result<(), VectorStoreError> {
        (**self).close().await
    }
}

#[async_trait]
impl<T> VectorRepository for Arc<T>
where
    T: VectorRepository + ?Sized,
{
    async fn ensure_collection(&self, dimension: usize) -> Result<bool, VectorStoreError> {
        (**self).ensure_collection(dimension).await
    }

    async fn collection_dimension(&self) -> Result<Option<usize>, VectorStoreError> {
        (**self).collection_dimension().await
    }

    async fn recreate_collection(&self, dimension: usize) -> Result<(), VectorStoreError> {
        (**self).recreate_collection(dimension).await
    }

    async fn upsert_vectors(
        &self,
        vectors: &[Vec<f32>],
        payloads: &[NotePayload],
        sparse_vectors: Option<&[SparseVector]>,
    ) -> Result<usize, VectorStoreError> {
        (**self)
            .upsert_vectors(vectors, payloads, sparse_vectors)
            .await
    }

    async fn delete_vectors(&self, note_ids: &[i64]) -> Result<usize, VectorStoreError> {
        (**self).delete_vectors(note_ids).await
    }

    async fn get_existing_hashes(
        &self,
        note_ids: &[i64],
    ) -> Result<HashMap<i64, String>, VectorStoreError> {
        (**self).get_existing_hashes(note_ids).await
    }

    async fn search_chunks(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SemanticSearchHit>, VectorStoreError> {
        (**self)
            .search_chunks(query_vector, query_sparse, limit, filters)
            .await
    }

    async fn search(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<(i64, f32)>, VectorStoreError> {
        (**self)
            .search(query_vector, query_sparse, limit, filters)
            .await
    }

    async fn find_similar_to_note(
        &self,
        note_id: i64,
        limit: usize,
        min_score: f32,
        deck_names: Option<&[String]>,
        tags: Option<&[String]>,
    ) -> Result<Vec<(i64, f32)>, VectorStoreError> {
        (**self)
            .find_similar_to_note(note_id, limit, min_score, deck_names, tags)
            .await
    }

    async fn close(&self) -> Result<(), VectorStoreError> {
        (**self).close().await
    }
}

/// Concrete Qdrant implementation.
pub struct QdrantRepository {
    _url: String,
    _collection_name: String,
}

impl QdrantRepository {
    /// Connect to Qdrant and create repository.
    pub async fn new(url: &str, collection_name: &str) -> Result<Self, VectorStoreError> {
        let grpc_url = common::config::qdrant_grpc_url(url)
            .map_err(|error| VectorStoreError::Connection(error.to_string()))?;

        // Validate URL by parsing it
        reqwest::Url::parse(&grpc_url)
            .map_err(|e| VectorStoreError::Connection(format!("invalid URL: {e}")))?;

        // Try to connect to validate the URL is reachable
        let _client = qdrant_client::Qdrant::from_url(&grpc_url)
            .build()
            .map_err(|e| VectorStoreError::Connection(e.to_string()))?;

        // Attempt a health check to verify connectivity
        _client
            .health_check()
            .await
            .map_err(|e| VectorStoreError::Connection(e.to_string()))?;

        Ok(Self {
            _url: grpc_url,
            _collection_name: collection_name.to_string(),
        })
    }

    /// Convert text into a hashed sparse vector (sha256 tokens, L2-normalized TF weights).
    pub fn text_to_sparse_vector(text: &str) -> SparseVector {
        // Tokenize: lowercase, keep only alphanumeric tokens
        let mut token_counts: HashMap<u32, f32> = HashMap::new();

        for token in text.split_whitespace() {
            let cleaned: String = token.chars().filter(|c| c.is_alphanumeric()).collect();
            let cleaned = cleaned.to_lowercase();
            if cleaned.is_empty() {
                continue;
            }

            // Hash token to u32 index using sha256
            let mut hasher = Sha256::new();
            hasher.update(cleaned.as_bytes());
            let hash = hasher.finalize();
            let index = u32::from_le_bytes([hash[0], hash[1], hash[2], hash[3]]);

            *token_counts.entry(index).or_insert(0.0) += 1.0;
        }

        if token_counts.is_empty() {
            return SparseVector::default();
        }

        // Compute TF weights: 1.0 + ln(count)
        let mut pairs: Vec<(u32, f32)> = token_counts
            .into_iter()
            .map(|(idx, count)| (idx, 1.0 + count.ln()))
            .collect();

        // Sort by index
        pairs.sort_by_key(|(idx, _)| *idx);

        // L2 normalize
        let norm: f32 = pairs.iter().map(|(_, v)| v * v).sum::<f32>().sqrt();

        let (indices, values): (Vec<u32>, Vec<f32>) = if norm > 0.0 {
            pairs.into_iter().map(|(i, v)| (i, v / norm)).unzip()
        } else {
            pairs.into_iter().unzip()
        };

        SparseVector { indices, values }
    }
}

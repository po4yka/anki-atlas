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

/// A note ID paired with its similarity score, returned from vector search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredNote {
    pub note_id: i64,
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

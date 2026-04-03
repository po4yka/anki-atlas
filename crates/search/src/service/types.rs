use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::fts::LexicalMode;
use crate::fusion::{FusionStats, SearchResult};

/// Controls which retrieval sources are used during search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    #[default]
    Hybrid,
    SemanticOnly,
    FtsOnly,
}

/// Parameters for a hybrid search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    pub query: String,
    pub filters: Option<crate::fts::SearchFilters>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_weight")]
    pub semantic_weight: f64,
    #[serde(default = "default_weight")]
    pub fts_weight: f64,
    #[serde(default)]
    pub search_mode: SearchMode,
    pub rerank_override: Option<bool>,
    pub rerank_top_n_override: Option<usize>,
}

/// Parameters for semantic chunk search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkSearchParams {
    pub query: String,
    pub filters: Option<crate::fts::SearchFilters>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

impl Default for ChunkSearchParams {
    fn default() -> Self {
        Self {
            query: String::new(),
            filters: None,
            limit: default_limit(),
        }
    }
}

pub(crate) fn default_limit() -> usize {
    50
}
pub(crate) fn default_weight() -> f64 {
    1.0
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            query: String::new(),
            filters: None,
            limit: default_limit(),
            semantic_weight: default_weight(),
            fts_weight: default_weight(),
            search_mode: SearchMode::Hybrid,
            rerank_override: None,
            rerank_top_n_override: None,
        }
    }
}

/// Complete hybrid search result.
#[derive(Debug, Clone, Serialize)]
pub struct HybridSearchResult {
    pub results: Vec<SearchResult>,
    pub stats: FusionStats,
    pub query: String,
    pub filters_applied: HashMap<String, serde_json::Value>,
    pub lexical_mode: LexicalMode,
    pub lexical_fallback_used: bool,
    pub query_suggestions: Vec<String>,
    pub autocomplete_suggestions: Vec<String>,
    pub rerank_applied: bool,
    pub rerank_model: Option<String>,
    pub rerank_top_n: Option<usize>,
}

impl HybridSearchResult {
    /// Create an empty result for short-circuit returns (e.g. blank queries).
    pub fn empty(query: String) -> Self {
        Self {
            results: vec![],
            stats: FusionStats::default(),
            query,
            filters_applied: HashMap::new(),
            lexical_mode: LexicalMode::None,
            lexical_fallback_used: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
            rerank_applied: false,
            rerank_model: None,
            rerank_top_n: None,
        }
    }
}

/// Single semantic chunk hit.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkSearchHit {
    pub note_id: i64,
    pub chunk_id: String,
    pub chunk_kind: String,
    pub modality: String,
    pub source_field: Option<String>,
    pub asset_rel_path: Option<String>,
    pub mime_type: Option<String>,
    pub preview_label: Option<String>,
    pub score: f64,
}

/// Semantic-only chunk search result.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkSearchResult {
    pub query: String,
    pub results: Vec<ChunkSearchHit>,
}

/// Detailed note information for result enrichment.
#[derive(Debug, Clone, Serialize)]
pub struct NoteDetail {
    pub note_id: i64,
    pub model_id: i64,
    pub normalized_text: String,
    pub tags: Vec<String>,
    pub deck_names: Vec<String>,
    pub mature: bool,
    pub lapses: i32,
    pub reps: i32,
}

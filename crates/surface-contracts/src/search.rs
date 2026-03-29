use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

fn default_limit() -> usize {
    50
}

fn default_weight() -> f64 {
    1.0
}

fn normalize_strings(values: Option<Vec<String>>) -> Option<Vec<String>> {
    values.and_then(|items| {
        let normalized = items
            .into_iter()
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect::<Vec<_>>();
        (!normalized.is_empty()).then_some(normalized)
    })
}

fn normalize_i64s(values: Option<Vec<i64>>) -> Option<Vec<i64>> {
    values.and_then(|items| (!items.is_empty()).then_some(items))
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct SearchFilterInput {
    pub deck_names: Option<Vec<String>>,
    pub deck_names_exclude: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub tags_exclude: Option<Vec<String>>,
    pub model_ids: Option<Vec<i64>>,
    pub min_ivl: Option<i32>,
    pub max_lapses: Option<i32>,
    pub min_reps: Option<i32>,
}

impl SearchFilterInput {
    pub fn normalized(self) -> Option<Self> {
        let normalized = Self {
            deck_names: normalize_strings(self.deck_names),
            deck_names_exclude: normalize_strings(self.deck_names_exclude),
            tags: normalize_strings(self.tags),
            tags_exclude: normalize_strings(self.tags_exclude),
            model_ids: normalize_i64s(self.model_ids),
            min_ivl: self.min_ivl,
            max_lapses: self.max_lapses,
            min_reps: self.min_reps,
        };

        let is_empty = normalized.deck_names.is_none()
            && normalized.deck_names_exclude.is_none()
            && normalized.tags.is_none()
            && normalized.tags_exclude.is_none()
            && normalized.model_ids.is_none()
            && normalized.min_ivl.is_none()
            && normalized.max_lapses.is_none()
            && normalized.min_reps.is_none();

        (!is_empty).then_some(normalized)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default)]
    pub filters: Option<SearchFilterInput>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_weight")]
    pub semantic_weight: f64,
    #[serde(default = "default_weight")]
    pub fts_weight: f64,
    #[serde(default)]
    pub semantic_only: bool,
    #[serde(default)]
    pub fts_only: bool,
    pub rerank_override: Option<bool>,
    pub rerank_top_n_override: Option<usize>,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            filters: None,
            limit: default_limit(),
            semantic_weight: default_weight(),
            fts_weight: default_weight(),
            semantic_only: false,
            fts_only: false,
            rerank_override: None,
            rerank_top_n_override: None,
        }
    }
}

impl SearchRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.limit == 0 {
            return Err("limit must be greater than 0".to_string());
        }
        if self.semantic_only && self.fts_only {
            return Err("semantic_only and fts_only cannot both be true".to_string());
        }
        if self.semantic_weight < 0.0 {
            return Err("semantic_weight must be non-negative".to_string());
        }
        if self.fts_weight < 0.0 {
            return Err("fts_weight must be non-negative".to_string());
        }
        if let Some(top_n) = self.rerank_top_n_override
            && top_n == 0
        {
            return Err("rerank_top_n_override must be greater than 0".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkSearchRequest {
    pub query: String,
    #[serde(default)]
    pub filters: Option<SearchFilterInput>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

impl Default for ChunkSearchRequest {
    fn default() -> Self {
        Self {
            query: String::new(),
            filters: None,
            limit: default_limit(),
        }
    }
}

impl ChunkSearchRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.limit == 0 {
            return Err("limit must be greater than 0".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LexicalMode {
    Fts,
    Fuzzy,
    Autocomplete,
    #[default]
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct FusionStats {
    pub semantic_only: usize,
    pub fts_only: usize,
    pub both: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SearchResultItem {
    pub note_id: i64,
    pub rrf_score: f64,
    pub semantic_score: Option<f64>,
    pub semantic_rank: Option<usize>,
    pub fts_score: Option<f64>,
    pub fts_rank: Option<usize>,
    pub headline: Option<String>,
    pub rerank_score: Option<f64>,
    pub rerank_rank: Option<usize>,
    pub sources: Vec<String>,
    pub match_modality: Option<String>,
    pub match_chunk_kind: Option<String>,
    pub match_source_field: Option<String>,
    pub match_asset_rel_path: Option<String>,
    pub match_preview_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SearchResponse {
    pub query: String,
    pub results: Vec<SearchResultItem>,
    pub stats: FusionStats,
    pub filters_applied: HashMap<String, Value>,
    pub lexical_mode: LexicalMode,
    pub lexical_fallback_used: bool,
    pub query_suggestions: Vec<String>,
    pub autocomplete_suggestions: Vec<String>,
    pub rerank_applied: bool,
    pub rerank_model: Option<String>,
    pub rerank_top_n: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
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

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ChunkSearchResponse {
    pub query: String,
    pub results: Vec<ChunkSearchHit>,
}

#[cfg(test)]
mod tests {
    use super::{
        ChunkSearchRequest, FusionStats, LexicalMode, SearchFilterInput, SearchRequest,
        SearchResponse, SearchResultItem,
    };

    #[test]
    fn search_filter_input_normalizes_strings_and_empties() {
        let normalized = SearchFilterInput {
            deck_names: Some(vec![" Rust ".to_string(), "".to_string()]),
            tags: Some(vec!["topic::ownership".to_string(), "   ".to_string()]),
            ..Default::default()
        }
        .normalized()
        .expect("expected normalized filters");

        assert_eq!(normalized.deck_names, Some(vec!["Rust".to_string()]));
        assert_eq!(normalized.tags, Some(vec!["topic::ownership".to_string()]));
    }

    #[test]
    fn search_request_validates_exclusive_modes() {
        let error = SearchRequest {
            semantic_only: true,
            fts_only: true,
            ..Default::default()
        }
        .validate()
        .expect_err("expected validation error");

        assert_eq!(error, "semantic_only and fts_only cannot both be true");
    }

    #[test]
    fn chunk_search_request_validates_limit() {
        let error = ChunkSearchRequest {
            limit: 0,
            ..Default::default()
        }
        .validate()
        .expect_err("expected validation error");

        assert_eq!(error, "limit must be greater than 0");
    }

    #[test]
    fn search_response_round_trips_through_json() {
        let response = SearchResponse {
            query: "ownership".to_string(),
            results: vec![SearchResultItem {
                note_id: 42,
                rrf_score: 1.25,
                semantic_score: Some(0.9),
                semantic_rank: Some(1),
                fts_score: Some(0.8),
                fts_rank: Some(2),
                headline: Some("Rust ownership".to_string()),
                rerank_score: Some(0.7),
                rerank_rank: Some(1),
                sources: vec!["semantic".to_string(), "fts".to_string()],
                match_modality: Some("text".to_string()),
                match_chunk_kind: Some("note".to_string()),
                match_source_field: Some("text".to_string()),
                match_asset_rel_path: None,
                match_preview_label: None,
            }],
            stats: FusionStats {
                semantic_only: 1,
                fts_only: 0,
                both: 1,
                total: 2,
            },
            lexical_mode: LexicalMode::Fts,
            ..Default::default()
        };

        let json = serde_json::to_string(&response).expect("serialize search response");
        let decoded: SearchResponse =
            serde_json::from_str(&json).expect("deserialize search response");

        assert_eq!(decoded, response);
    }
}

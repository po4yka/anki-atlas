use serde::{Deserialize, Serialize};

use crate::error::SearchError;

/// Source of a lexical search result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FtsSource {
    Fts,
    Fuzzy,
    Autocomplete,
}

/// Single FTS hit.
#[derive(Debug, Clone, Serialize)]
pub struct FtsResult {
    pub note_id: i64,
    pub rank: f64,
    pub headline: Option<String>,
    pub source: FtsSource,
}

/// Lexical search mode that was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalMode {
    Fts,
    Fuzzy,
    Autocomplete,
    None,
}

/// Bundle of lexical results with fallback metadata.
#[derive(Debug, Clone, Serialize)]
pub struct LexicalSearchResult {
    pub results: Vec<FtsResult>,
    pub mode: LexicalMode,
    pub used_fallback: bool,
    pub query_suggestions: Vec<String>,
    pub autocomplete_suggestions: Vec<String>,
}

/// Filters for search queries (shared between FTS and hybrid).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilters {
    pub deck_names: Option<Vec<String>>,
    pub deck_names_exclude: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub tags_exclude: Option<Vec<String>>,
    pub model_ids: Option<Vec<i64>>,
    pub min_ivl: Option<i32>,
    pub max_lapses: Option<i32>,
    pub min_reps: Option<i32>,
}

/// Execute lexical search with FTS -> fuzzy -> autocomplete fallback chain.
pub async fn search_lexical(
    _pool: &sqlx::PgPool,
    _query: &str,
    _filters: Option<&SearchFilters>,
    _limit: i64,
) -> Result<LexicalSearchResult, SearchError> {
    todo!("search_lexical not yet implemented")
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Type construction tests ---

    #[test]
    fn fts_source_variants_exist() {
        let sources = [FtsSource::Fts, FtsSource::Fuzzy, FtsSource::Autocomplete];
        assert_eq!(sources.len(), 3);
    }

    #[test]
    fn fts_source_serde_roundtrip() {
        let source = FtsSource::Fts;
        let json = serde_json::to_string(&source).unwrap();
        assert_eq!(json, "\"fts\"");

        let fuzzy: FtsSource = serde_json::from_str("\"fuzzy\"").unwrap();
        assert_eq!(fuzzy, FtsSource::Fuzzy);

        let auto: FtsSource = serde_json::from_str("\"autocomplete\"").unwrap();
        assert_eq!(auto, FtsSource::Autocomplete);
    }

    #[test]
    fn lexical_mode_serde_roundtrip() {
        let mode = LexicalMode::None;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"none\"");

        let fts: LexicalMode = serde_json::from_str("\"fts\"").unwrap();
        assert_eq!(fts, LexicalMode::Fts);

        let fuzzy: LexicalMode = serde_json::from_str("\"fuzzy\"").unwrap();
        assert_eq!(fuzzy, LexicalMode::Fuzzy);

        let auto: LexicalMode = serde_json::from_str("\"autocomplete\"").unwrap();
        assert_eq!(auto, LexicalMode::Autocomplete);
    }

    #[test]
    fn fts_result_construction() {
        let result = FtsResult {
            note_id: 42,
            rank: 0.95,
            headline: Some("matched <b>term</b>".to_string()),
            source: FtsSource::Fts,
        };
        assert_eq!(result.note_id, 42);
        assert!((result.rank - 0.95).abs() < f64::EPSILON);
        assert_eq!(result.headline.as_deref(), Some("matched <b>term</b>"));
        assert_eq!(result.source, FtsSource::Fts);
    }

    #[test]
    fn fts_result_without_headline() {
        let result = FtsResult {
            note_id: 1,
            rank: 0.5,
            headline: None,
            source: FtsSource::Autocomplete,
        };
        assert!(result.headline.is_none());
        assert_eq!(result.source, FtsSource::Autocomplete);
    }

    #[test]
    fn lexical_search_result_construction() {
        let result = LexicalSearchResult {
            results: vec![
                FtsResult {
                    note_id: 1,
                    rank: 0.9,
                    headline: Some("hl".to_string()),
                    source: FtsSource::Fts,
                },
                FtsResult {
                    note_id: 2,
                    rank: 0.8,
                    headline: None,
                    source: FtsSource::Fts,
                },
            ],
            mode: LexicalMode::Fts,
            used_fallback: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
        };
        assert_eq!(result.results.len(), 2);
        assert_eq!(result.mode, LexicalMode::Fts);
        assert!(!result.used_fallback);
        assert!(result.query_suggestions.is_empty());
        assert!(result.autocomplete_suggestions.is_empty());
    }

    #[test]
    fn lexical_search_result_with_fallback() {
        let result = LexicalSearchResult {
            results: vec![FtsResult {
                note_id: 10,
                rank: 0.3,
                headline: None,
                source: FtsSource::Fuzzy,
            }],
            mode: LexicalMode::Fuzzy,
            used_fallback: true,
            query_suggestions: vec!["suggestion1".to_string(), "suggestion2".to_string()],
            autocomplete_suggestions: vec!["auto1".to_string()],
        };
        assert_eq!(result.mode, LexicalMode::Fuzzy);
        assert!(result.used_fallback);
        assert_eq!(result.query_suggestions.len(), 2);
        assert_eq!(result.autocomplete_suggestions.len(), 1);
    }

    #[test]
    fn lexical_search_result_none_mode() {
        let result = LexicalSearchResult {
            results: vec![],
            mode: LexicalMode::None,
            used_fallback: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
        };
        assert!(result.results.is_empty());
        assert_eq!(result.mode, LexicalMode::None);
    }

    // --- SearchFilters tests ---

    #[test]
    fn search_filters_default_all_none() {
        let filters = SearchFilters::default();
        assert!(filters.deck_names.is_none());
        assert!(filters.deck_names_exclude.is_none());
        assert!(filters.tags.is_none());
        assert!(filters.tags_exclude.is_none());
        assert!(filters.model_ids.is_none());
        assert!(filters.min_ivl.is_none());
        assert!(filters.max_lapses.is_none());
        assert!(filters.min_reps.is_none());
    }

    #[test]
    fn search_filters_serde_roundtrip() {
        let filters = SearchFilters {
            deck_names: Some(vec!["Default".to_string()]),
            deck_names_exclude: None,
            tags: Some(vec!["rust".to_string(), "programming".to_string()]),
            tags_exclude: Some(vec!["deprecated".to_string()]),
            model_ids: Some(vec![123, 456]),
            min_ivl: Some(21),
            max_lapses: Some(5),
            min_reps: Some(3),
        };
        let json = serde_json::to_string(&filters).unwrap();
        let deserialized: SearchFilters = serde_json::from_str(&json).unwrap();

        assert_eq!(
            deserialized.deck_names.as_ref().unwrap(),
            &["Default".to_string()]
        );
        assert!(deserialized.deck_names_exclude.is_none());
        assert_eq!(deserialized.tags.as_ref().unwrap().len(), 2);
        assert_eq!(deserialized.model_ids.as_ref().unwrap(), &[123, 456]);
        assert_eq!(deserialized.min_ivl, Some(21));
        assert_eq!(deserialized.max_lapses, Some(5));
        assert_eq!(deserialized.min_reps, Some(3));
    }

    #[test]
    fn search_filters_partial_construction() {
        let filters = SearchFilters {
            tags: Some(vec!["math".to_string()]),
            min_ivl: Some(7),
            ..SearchFilters::default()
        };
        assert_eq!(filters.tags.as_ref().unwrap(), &["math".to_string()]);
        assert_eq!(filters.min_ivl, Some(7));
        assert!(filters.deck_names.is_none());
        assert!(filters.model_ids.is_none());
    }

    // --- search_lexical behavior tests ---

    #[tokio::test]
    async fn search_lexical_blank_query_returns_none_mode() {
        // Blank queries should return immediately with mode=None, no DB access needed.
        // Use a dummy pool options to create a pool that won't connect.
        let pool = sqlx::PgPool::connect_lazy("postgres://invalid:5432/fake").unwrap();
        let result = search_lexical(&pool, "", None, 10).await.unwrap();
        assert_eq!(result.mode, LexicalMode::None);
        assert!(result.results.is_empty());
        assert!(!result.used_fallback);
        assert!(result.query_suggestions.is_empty());
        assert!(result.autocomplete_suggestions.is_empty());
    }

    #[tokio::test]
    async fn search_lexical_whitespace_query_returns_none_mode() {
        let pool = sqlx::PgPool::connect_lazy("postgres://invalid:5432/fake").unwrap();
        let result = search_lexical(&pool, "   ", None, 10).await.unwrap();
        assert_eq!(result.mode, LexicalMode::None);
        assert!(result.results.is_empty());
        assert!(!result.used_fallback);
    }

    // --- FtsResult serialization ---

    #[test]
    fn fts_result_serializes_to_json() {
        let result = FtsResult {
            note_id: 99,
            rank: 0.75,
            headline: Some("test headline".to_string()),
            source: FtsSource::Fuzzy,
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["note_id"], 99);
        assert_eq!(json["source"], "fuzzy");
        assert_eq!(json["headline"], "test headline");
    }

    #[test]
    fn lexical_search_result_serializes_to_json() {
        let result = LexicalSearchResult {
            results: vec![],
            mode: LexicalMode::Autocomplete,
            used_fallback: true,
            query_suggestions: vec!["fix".to_string()],
            autocomplete_suggestions: vec!["fixing".to_string(), "fixed".to_string()],
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["mode"], "autocomplete");
        assert_eq!(json["used_fallback"], true);
        assert_eq!(json["query_suggestions"].as_array().unwrap().len(), 1);
        assert_eq!(
            json["autocomplete_suggestions"].as_array().unwrap().len(),
            2
        );
    }

    // --- FtsSource equality + clone ---

    #[test]
    fn fts_source_clone_and_eq() {
        let a = FtsSource::Fts;
        let b = a;
        assert_eq!(a, b);

        let c = FtsSource::Fuzzy;
        assert_ne!(a, c);
    }

    #[test]
    fn lexical_mode_clone_and_eq() {
        let a = LexicalMode::Fts;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(LexicalMode::Fts, LexicalMode::Fuzzy);
        assert_ne!(LexicalMode::Fuzzy, LexicalMode::Autocomplete);
        assert_ne!(LexicalMode::Autocomplete, LexicalMode::None);
    }
}

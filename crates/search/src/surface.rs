use crate::fts::SearchFilters;
use crate::service::{ChunkSearchParams, SearchParams};

#[derive(Debug, Clone, Default, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
pub struct SearchRequestInput {
    pub query: String,
    pub filters: SearchFilterInput,
    pub limit: usize,
    pub semantic_weight: f64,
    pub fts_weight: f64,
    pub semantic_only: bool,
    pub fts_only: bool,
    pub rerank_override: Option<bool>,
    pub rerank_top_n_override: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkSearchRequestInput {
    pub query: String,
    pub filters: SearchFilterInput,
    pub limit: usize,
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

pub fn build_search_filters(input: SearchFilterInput) -> Option<SearchFilters> {
    let filters = SearchFilters {
        deck_names: normalize_strings(input.deck_names),
        deck_names_exclude: normalize_strings(input.deck_names_exclude),
        tags: normalize_strings(input.tags),
        tags_exclude: normalize_strings(input.tags_exclude),
        model_ids: normalize_i64s(input.model_ids),
        min_ivl: input.min_ivl,
        max_lapses: input.max_lapses,
        min_reps: input.min_reps,
    };

    let is_empty = filters.deck_names.is_none()
        && filters.deck_names_exclude.is_none()
        && filters.tags.is_none()
        && filters.tags_exclude.is_none()
        && filters.model_ids.is_none()
        && filters.min_ivl.is_none()
        && filters.max_lapses.is_none()
        && filters.min_reps.is_none();

    (!is_empty).then_some(filters)
}

pub fn build_search_params(input: SearchRequestInput) -> Result<SearchParams, String> {
    if input.limit == 0 {
        return Err("limit must be greater than 0".to_string());
    }
    if input.semantic_only && input.fts_only {
        return Err("semantic_only and fts_only cannot both be true".to_string());
    }
    if input.semantic_weight < 0.0 {
        return Err("semantic_weight must be non-negative".to_string());
    }
    if input.fts_weight < 0.0 {
        return Err("fts_weight must be non-negative".to_string());
    }
    if let Some(top_n) = input.rerank_top_n_override
        && top_n == 0
    {
        return Err("rerank_top_n_override must be greater than 0".to_string());
    }

    Ok(SearchParams {
        query: input.query,
        filters: build_search_filters(input.filters),
        limit: input.limit,
        semantic_weight: input.semantic_weight,
        fts_weight: input.fts_weight,
        semantic_only: input.semantic_only,
        fts_only: input.fts_only,
        rerank_override: input.rerank_override,
        rerank_top_n_override: input.rerank_top_n_override,
    })
}

pub fn build_chunk_search_params(
    input: ChunkSearchRequestInput,
) -> Result<ChunkSearchParams, String> {
    if input.limit == 0 {
        return Err("limit must be greater than 0".to_string());
    }

    Ok(ChunkSearchParams {
        query: input.query,
        filters: build_search_filters(input.filters),
        limit: input.limit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_search_filters_strips_empty_values() {
        let filters = build_search_filters(SearchFilterInput {
            deck_names: Some(vec![" Deck ".to_string(), "".to_string()]),
            tags: Some(vec!["rust".to_string(), "   ".to_string()]),
            ..Default::default()
        })
        .expect("filters should be present");

        assert_eq!(filters.deck_names, Some(vec!["Deck".to_string()]));
        assert_eq!(filters.tags, Some(vec!["rust".to_string()]));
    }

    #[test]
    fn build_search_filters_returns_none_for_empty_input() {
        assert!(build_search_filters(SearchFilterInput::default()).is_none());
    }

    #[test]
    fn build_search_params_validates_exclusive_modes() {
        let error = build_search_params(SearchRequestInput {
            query: "rust".to_string(),
            filters: SearchFilterInput::default(),
            limit: 10,
            semantic_weight: 1.0,
            fts_weight: 1.0,
            semantic_only: true,
            fts_only: true,
            rerank_override: None,
            rerank_top_n_override: None,
        })
        .expect_err("mutually exclusive flags should fail");

        assert_eq!(error, "semantic_only and fts_only cannot both be true");
    }

    #[test]
    fn build_chunk_search_params_validates_limit() {
        let error = build_chunk_search_params(ChunkSearchRequestInput {
            query: "rust".to_string(),
            filters: SearchFilterInput::default(),
            limit: 0,
        })
        .expect_err("zero limit should fail");

        assert_eq!(error, "limit must be greater than 0");
    }
}

use analytics::coverage::{
    GapType, TopicCoverage as AnalyticsTopicCoverage, TopicGap as AnalyticsTopicGap,
    WeakNote as AnalyticsWeakNote,
};
use analytics::duplicates::{
    DuplicateCluster as AnalyticsDuplicateCluster, DuplicateDetail as AnalyticsDuplicateDetail,
    DuplicateStats as AnalyticsDuplicateStats,
};
use analytics::labeling::LabelingStats as AnalyticsLabelingStats;
use analytics::taxonomy::Taxonomy;
use search::error::SearchError;
use search::fts::SearchFilters;
use search::fusion::{FusionStats as SearchFusionStats, SearchResult};
use search::service::{
    ChunkSearchHit as SearchChunkSearchHit, ChunkSearchParams, ChunkSearchResult,
    HybridSearchResult, SearchParams,
};
use surface_contracts::analytics::{
    DuplicateCluster, DuplicateDetail, DuplicateStats, GapKind, LabelingStats, TaxonomyLoadSummary,
    TopicCoverage, TopicGap, WeakNote,
};
use surface_contracts::search::{
    ChunkSearchHit, ChunkSearchRequest, ChunkSearchResponse,
    FusionStats, LexicalMode, SearchFilterInput, SearchRequest, SearchResponse, SearchResultItem,
};

pub fn build_search_params(request: &SearchRequest) -> Result<SearchParams, SearchError> {
    request.validate().map_err(SearchError::InvalidRequest)?;

    Ok(SearchParams {
        query: request.query.clone(),
        filters: build_search_filters(request.filters.clone()),
        limit: request.limit,
        semantic_weight: request.semantic_weight,
        fts_weight: request.fts_weight,
        semantic_only: request.semantic_only,
        fts_only: request.fts_only,
        rerank_override: request.rerank_override,
        rerank_top_n_override: request.rerank_top_n_override,
    })
}

pub fn build_chunk_search_params(
    request: &ChunkSearchRequest,
) -> Result<ChunkSearchParams, SearchError> {
    request.validate().map_err(SearchError::InvalidRequest)?;

    Ok(ChunkSearchParams {
        query: request.query.clone(),
        filters: build_search_filters(request.filters.clone()),
        limit: request.limit,
    })
}

fn build_search_filters(input: Option<SearchFilterInput>) -> Option<SearchFilters> {
    let normalized = input.and_then(|filters| filters.normalized())?;

    Some(SearchFilters {
        deck_names: normalized.deck_names,
        deck_names_exclude: normalized.deck_names_exclude,
        tags: normalized.tags,
        tags_exclude: normalized.tags_exclude,
        model_ids: normalized.model_ids,
        min_ivl: normalized.min_ivl,
        max_lapses: normalized.max_lapses,
        min_reps: normalized.min_reps,
    })
}

pub fn taxonomy_load_summary(taxonomy: &Taxonomy) -> TaxonomyLoadSummary {
    TaxonomyLoadSummary {
        topic_count: taxonomy.topics.len(),
        root_count: taxonomy.roots.len(),
    }
}

fn lexical_mode(value: search::fts::LexicalMode) -> LexicalMode {
    match value {
        search::fts::LexicalMode::Fts => LexicalMode::Fts,
        search::fts::LexicalMode::Fuzzy => LexicalMode::Fuzzy,
        search::fts::LexicalMode::Autocomplete => LexicalMode::Autocomplete,
        search::fts::LexicalMode::None => LexicalMode::None,
    }
}

fn fusion_stats(value: SearchFusionStats) -> FusionStats {
    FusionStats {
        semantic_only: value.semantic_only,
        fts_only: value.fts_only,
        both: value.both,
        total: value.total,
    }
}

fn search_result_item(value: SearchResult) -> SearchResultItem {
    let sources = value.sources().into_iter().map(str::to_string).collect();

    SearchResultItem {
        note_id: value.note_id,
        rrf_score: value.rrf_score,
        semantic_score: value.semantic_score,
        semantic_rank: value.semantic_rank,
        fts_score: value.fts_score,
        fts_rank: value.fts_rank,
        headline: value.headline,
        rerank_score: value.rerank_score,
        rerank_rank: value.rerank_rank,
        sources,
        match_modality: value.match_modality,
        match_chunk_kind: value.match_chunk_kind,
        match_source_field: value.match_source_field,
        match_asset_rel_path: value.match_asset_rel_path,
        match_preview_label: value.match_preview_label,
    }
}

pub fn search_response(value: HybridSearchResult) -> SearchResponse {
    SearchResponse {
        query: value.query,
        results: value.results.into_iter().map(search_result_item).collect(),
        stats: fusion_stats(value.stats),
        filters_applied: value.filters_applied,
        lexical_mode: lexical_mode(value.lexical_mode),
        lexical_fallback_used: value.lexical_fallback_used,
        query_suggestions: value.query_suggestions,
        autocomplete_suggestions: value.autocomplete_suggestions,
        rerank_applied: value.rerank_applied,
        rerank_model: value.rerank_model,
        rerank_top_n: value.rerank_top_n,
    }
}

fn chunk_search_hit(value: SearchChunkSearchHit) -> ChunkSearchHit {
    ChunkSearchHit {
        note_id: value.note_id,
        chunk_id: value.chunk_id,
        chunk_kind: value.chunk_kind,
        modality: value.modality,
        source_field: value.source_field,
        asset_rel_path: value.asset_rel_path,
        mime_type: value.mime_type,
        preview_label: value.preview_label,
        score: value.score,
    }
}

pub fn chunk_search_response(value: ChunkSearchResult) -> ChunkSearchResponse {
    ChunkSearchResponse {
        query: value.query,
        results: value.results.into_iter().map(chunk_search_hit).collect(),
    }
}

fn gap_kind(value: GapType) -> GapKind {
    match value {
        GapType::Missing => GapKind::Missing,
        GapType::Undercovered => GapKind::Undercovered,
    }
}

pub fn topic_coverage(value: AnalyticsTopicCoverage) -> TopicCoverage {
    TopicCoverage {
        topic_id: value.topic_id,
        path: value.path,
        label: value.label,
        note_count: value.note_count,
        subtree_count: value.subtree_count,
        child_count: value.child_count,
        covered_children: value.covered_children,
        mature_count: value.mature_count,
        avg_confidence: value.avg_confidence,
        weak_notes: value.weak_notes,
        avg_lapses: value.avg_lapses,
    }
}

fn topic_gap(value: AnalyticsTopicGap) -> TopicGap {
    TopicGap {
        topic_id: value.topic_id,
        path: value.path,
        label: value.label,
        description: value.description,
        gap_type: gap_kind(value.gap_type),
        note_count: value.note_count,
        threshold: value.threshold,
        nearest_notes: value.nearest_notes,
    }
}

fn weak_note(value: AnalyticsWeakNote) -> WeakNote {
    WeakNote {
        note_id: value.note_id,
        topic_path: value.topic_path,
        confidence: value.confidence,
        lapses: value.lapses,
        fail_rate: value.fail_rate,
        normalized_text: value.normalized_text,
    }
}

fn duplicate_detail(value: AnalyticsDuplicateDetail) -> DuplicateDetail {
    DuplicateDetail {
        note_id: value.note_id,
        similarity: value.similarity,
        text: value.text,
        deck_names: value.deck_names,
        tags: value.tags,
    }
}

fn duplicate_cluster(value: AnalyticsDuplicateCluster) -> DuplicateCluster {
    DuplicateCluster {
        representative_id: value.representative_id,
        representative_text: value.representative_text,
        duplicates: value.duplicates.into_iter().map(duplicate_detail).collect(),
        deck_names: value.deck_names,
        tags: value.tags,
    }
}

fn duplicate_stats(value: AnalyticsDuplicateStats) -> DuplicateStats {
    DuplicateStats {
        notes_scanned: value.notes_scanned,
        clusters_found: value.clusters_found,
        total_duplicates: value.total_duplicates,
        avg_cluster_size: value.avg_cluster_size,
    }
}

pub fn labeling_stats(value: AnalyticsLabelingStats) -> LabelingStats {
    LabelingStats {
        notes_processed: value.notes_processed,
        assignments_created: value.assignments_created,
        topics_matched: value.topics_matched,
    }
}

pub fn topic_gaps(values: Vec<AnalyticsTopicGap>) -> Vec<TopicGap> {
    values.into_iter().map(topic_gap).collect()
}

pub fn weak_notes(values: Vec<AnalyticsWeakNote>) -> Vec<WeakNote> {
    values.into_iter().map(weak_note).collect()
}

pub fn duplicates(
    clusters: Vec<AnalyticsDuplicateCluster>,
    stats: AnalyticsDuplicateStats,
) -> (Vec<DuplicateCluster>, DuplicateStats) {
    (
        clusters.into_iter().map(duplicate_cluster).collect(),
        duplicate_stats(stats),
    )
}

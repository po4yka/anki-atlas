use analytics::coverage::{GapType, TopicCoverage, TopicGap, WeakNote};
use analytics::duplicates::{DuplicateCluster, DuplicateDetail, DuplicateStats};
use anki_atlas_api::schemas::*;
use jobs::types::{JobStatus, JobType};
use search::fts::LexicalMode;
use search::fusion::{FusionStats, SearchResult};
use search::service::{HybridSearchResult, SearchParams};
use serde_json::{Value, json};
use std::collections::HashMap;

#[test]
fn async_sync_request_defaults() {
    let json = json!({ "source": "/path/col.anki2" });
    let req: AsyncSyncRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.source, "/path/col.anki2");
    assert!(req.run_migrations);
    assert!(req.index);
    assert!(!req.force_reindex);
    assert!(req.run_at.is_none());
}

#[test]
fn async_index_request_defaults() {
    let req: AsyncIndexRequest = serde_json::from_value(json!({})).unwrap();
    assert!(!req.force_reindex);
    assert!(req.run_at.is_none());
}

#[test]
fn job_status_response_serializes() {
    let resp = JobStatusResponse {
        job_id: "job-456".into(),
        job_type: JobType::Index,
        status: JobStatus::Running,
        progress: 0.5,
        message: Some("halfway".into()),
        attempts: 1,
        max_retries: 3,
        cancel_requested: false,
        created_at: Some(chrono::Utc::now()),
        scheduled_for: None,
        started_at: Some(chrono::Utc::now()),
        finished_at: None,
        result: None,
        error: None,
    };
    let v: Value = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["progress"], 0.5);
    assert_eq!(v["message"], "halfway");
}

#[test]
fn search_request_deserializes_nested_filters() {
    let req: SearchRequest = serde_json::from_value(json!({
        "query": "ownership",
        "filters": {
            "deck_names": ["Rust"],
            "tags": ["ownership"]
        },
        "limit": 10,
        "semantic_only": true,
        "rerank_override": true,
        "rerank_top_n_override": 5
    }))
    .unwrap();

    assert_eq!(req.query, "ownership");
    assert_eq!(req.limit, 10);
    assert!(req.semantic_only);
    assert_eq!(req.rerank_override, Some(true));
    assert_eq!(req.rerank_top_n_override, Some(5));
    assert_eq!(
        req.filters.unwrap().deck_names.unwrap(),
        vec!["Rust".to_string()]
    );
}

#[test]
fn search_request_validation_rejects_conflicting_modes() {
    let req = SearchRequest {
        query: "ownership".into(),
        filters: None,
        limit: 50,
        semantic_weight: 1.0,
        fts_weight: 1.0,
        semantic_only: true,
        fts_only: true,
        rerank_override: None,
        rerank_top_n_override: None,
    };
    assert!(req.validate().is_err());
}

#[test]
fn search_request_converts_to_search_params() {
    let request = SearchRequest {
        query: "lifetimes".into(),
        filters: Some(SearchFiltersDto {
            deck_names: Some(vec!["Rust".into()]),
            model_ids: Some(vec![1001]),
            ..Default::default()
        }),
        limit: 15,
        semantic_weight: 0.8,
        fts_weight: 0.2,
        semantic_only: false,
        fts_only: true,
        rerank_override: Some(false),
        rerank_top_n_override: Some(3),
    };

    let params: SearchParams = request.into();
    assert_eq!(params.query, "lifetimes");
    assert_eq!(params.limit, 15);
    assert!(params.fts_only);
    assert_eq!(params.rerank_override, Some(false));
    let filters = params.filters.expect("filters");
    assert_eq!(filters.deck_names, Some(vec!["Rust".into()]));
    assert_eq!(filters.deck_names_exclude, None);
    assert_eq!(filters.tags, None);
    assert_eq!(filters.tags_exclude, None);
    assert_eq!(filters.model_ids, Some(vec![1001]));
    assert_eq!(filters.min_ivl, None);
    assert_eq!(filters.max_lapses, None);
    assert_eq!(filters.min_reps, None);
}

#[test]
fn search_response_serializes_typed_metadata() {
    let response = SearchResponse::from(HybridSearchResult {
        query: "ownership".into(),
        results: vec![SearchResult {
            note_id: 1,
            rrf_score: 0.95,
            semantic_score: Some(0.9),
            semantic_rank: Some(1),
            fts_score: Some(0.8),
            fts_rank: Some(2),
            headline: Some("headline".into()),
            rerank_score: Some(0.97),
            rerank_rank: Some(1),
            match_modality: Some("text".into()),
            match_chunk_kind: Some("text_primary".into()),
            match_source_field: None,
            match_asset_rel_path: None,
            match_preview_label: Some("headline".into()),
        }],
        stats: FusionStats {
            semantic_only: 0,
            fts_only: 0,
            both: 1,
            total: 1,
        },
        filters_applied: HashMap::new(),
        lexical_mode: LexicalMode::Fts,
        lexical_fallback_used: false,
        query_suggestions: vec!["ownership borrowing".into()],
        autocomplete_suggestions: vec!["ownership".into()],
        rerank_applied: true,
        rerank_model: Some("cross-encoder/test".into()),
        rerank_top_n: Some(10),
    });

    let v: Value = serde_json::to_value(&response).unwrap();
    assert_eq!(v["lexical_mode"], "fts");
    assert_eq!(v["rerank_model"], "cross-encoder/test");
    assert_eq!(v["results"][0]["semantic_rank"], 1);
}

#[test]
fn topics_tree_response_serializes() {
    let response = TopicsTreeResponse {
        topics: vec![json!({"path": "cs", "label": "CS"})],
    };
    let v: Value = serde_json::to_value(&response).unwrap();
    assert_eq!(v["topics"][0]["path"], "cs");
}

#[test]
fn topic_coverage_response_maps_from_domain() {
    let response = TopicCoverageResponse::from(TopicCoverage {
        topic_id: 42,
        path: "cs/algorithms".into(),
        label: "Algorithms".into(),
        note_count: 4,
        subtree_count: 6,
        child_count: 2,
        covered_children: 1,
        mature_count: 2,
        avg_confidence: 0.8,
        weak_notes: 1,
        avg_lapses: 0.5,
    });
    let v: Value = serde_json::to_value(&response).unwrap();
    assert_eq!(v["topic_id"], 42);
    assert_eq!(v["label"], "Algorithms");
}

#[test]
fn topic_gaps_response_serializes_typed_gap_type() {
    let response = TopicGapsResponse {
        root_path: "cs".into(),
        min_coverage: 2,
        gaps: vec![TopicGapItem::from(TopicGap {
            topic_id: 10,
            path: "cs/networking".into(),
            label: "Networking".into(),
            description: None,
            gap_type: GapType::Missing,
            note_count: 0,
            threshold: 2,
            nearest_notes: vec![],
        })],
        missing_count: 1,
        undercovered_count: 0,
    };
    let v: Value = serde_json::to_value(&response).unwrap();
    assert_eq!(v["gaps"][0]["gap_type"], "missing");
}

#[test]
fn topic_weak_notes_response_serializes() {
    let response = TopicWeakNotesResponse {
        topic_path: "cs".into(),
        max_results: 20,
        notes: vec![TopicWeakNoteItem::from(WeakNote {
            note_id: 7,
            topic_path: "cs".into(),
            confidence: 0.7,
            lapses: 3,
            fail_rate: Some(0.2),
            normalized_text: "preview".into(),
        })],
    };
    let v: Value = serde_json::to_value(&response).unwrap();
    assert_eq!(v["notes"][0]["note_id"], 7);
}

#[test]
fn duplicates_query_parses_bracketed_filters() {
    let query = DuplicatesQuery::from_query_string(Some(
        "threshold=0.9&max_clusters=5&deck_filter[]=Rust&tag_filter[]=ownership",
    ))
    .expect("query");

    assert!((query.threshold - 0.9).abs() < f64::EPSILON);
    assert_eq!(query.max_clusters, 5);
    assert_eq!(query.deck_filter, Some(vec!["Rust".into()]));
    assert_eq!(query.tag_filter, Some(vec!["ownership".into()]));
}

#[test]
fn duplicates_response_maps_from_domain() {
    let response = DuplicatesResponse {
        clusters: vec![DuplicateClusterItem::from(DuplicateCluster {
            representative_id: 100,
            representative_text: "What is X?".into(),
            duplicates: vec![DuplicateDetail {
                note_id: 101,
                similarity: 0.95,
                text: "What is X?".into(),
                deck_names: vec!["Deck2".into()],
                tags: vec!["tag2".into()],
            }],
            deck_names: vec!["Deck1".into()],
            tags: vec!["tag1".into()],
        })],
        stats: DuplicateStatsResponse::from(DuplicateStats {
            notes_scanned: 2,
            clusters_found: 1,
            total_duplicates: 1,
            avg_cluster_size: 2.0,
        }),
    };
    let v: Value = serde_json::to_value(&response).unwrap();
    assert_eq!(v["clusters"][0]["size"], 2);
    assert_eq!(v["stats"]["clusters_found"], 1);
}

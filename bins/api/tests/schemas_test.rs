use anki_atlas_api::schemas::*;
use jobs::types::{JobStatus, JobType};
use serde_json::{Value, json};

// --- SyncRequest ---

#[test]
fn sync_request_deserializes_minimal() {
    let json = json!({ "source": "/path/to/collection.anki2" });
    let req: SyncRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.source, "/path/to/collection.anki2");
    assert!(req.run_migrations, "run_migrations should default to true");
    assert!(req.index, "index should default to true");
    assert!(!req.force_reindex, "force_reindex should default to false");
}

#[test]
fn sync_request_deserializes_all_fields() {
    let json = json!({
        "source": "/data/col.anki21",
        "run_migrations": false,
        "index": false,
        "force_reindex": true,
    });
    let req: SyncRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.source, "/data/col.anki21");
    assert!(!req.run_migrations);
    assert!(!req.index);
    assert!(req.force_reindex);
}

#[test]
fn sync_request_rejects_missing_source() {
    let json = json!({});
    let result = serde_json::from_value::<SyncRequest>(json);
    assert!(result.is_err());
}

// --- SyncResponse ---

#[test]
fn sync_response_serializes_complete() {
    let resp = SyncResponse {
        status: "completed".into(),
        decks_upserted: 5,
        models_upserted: 2,
        notes_upserted: 100,
        notes_deleted: 3,
        cards_upserted: 200,
        card_stats_upserted: 200,
        duration_ms: 1500,
        notes_embedded: Some(95),
        notes_skipped: Some(5),
        index_errors: Some(vec!["err1".into()]),
    };
    let v: Value = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["status"], "completed");
    assert_eq!(v["decks_upserted"], 5);
    assert_eq!(v["notes_embedded"], 95);
    assert_eq!(v["index_errors"][0], "err1");
}

#[test]
fn sync_response_serializes_nulls_for_none() {
    let resp = SyncResponse {
        status: "completed".into(),
        decks_upserted: 0,
        models_upserted: 0,
        notes_upserted: 0,
        notes_deleted: 0,
        cards_upserted: 0,
        card_stats_upserted: 0,
        duration_ms: 0,
        notes_embedded: None,
        notes_skipped: None,
        index_errors: None,
    };
    let v: Value = serde_json::to_value(&resp).unwrap();
    assert!(v["notes_embedded"].is_null());
    assert!(v["notes_skipped"].is_null());
    assert!(v["index_errors"].is_null());
}

// --- IndexRequest/Response ---

#[test]
fn index_request_defaults() {
    let json = json!({});
    let req: IndexRequest = serde_json::from_value(json).unwrap();
    assert!(!req.force_reindex);
}

#[test]
fn index_response_serializes() {
    let resp = IndexResponse {
        status: "completed".into(),
        notes_processed: 50,
        notes_embedded: 45,
        notes_skipped: 5,
        notes_deleted: 0,
        errors: vec![],
    };
    let v: Value = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["notes_processed"], 50);
    assert!(v["errors"].as_array().unwrap().is_empty());
}

// --- AsyncSyncRequest ---

#[test]
fn async_sync_request_with_run_at() {
    let json = json!({
        "source": "/path/col.anki2",
        "run_at": "2026-03-09T10:00:00Z",
    });
    let req: AsyncSyncRequest = serde_json::from_value(json).unwrap();
    assert!(req.run_at.is_some());
    assert!(req.run_migrations);
    assert!(req.index);
}

#[test]
fn async_sync_request_without_run_at() {
    let json = json!({ "source": "/path/col.anki2" });
    let req: AsyncSyncRequest = serde_json::from_value(json).unwrap();
    assert!(req.run_at.is_none());
}

// --- AsyncIndexRequest ---

#[test]
fn async_index_request_defaults() {
    let json = json!({});
    let req: AsyncIndexRequest = serde_json::from_value(json).unwrap();
    assert!(!req.force_reindex);
    assert!(req.run_at.is_none());
}

// --- JobAcceptedResponse ---

#[test]
fn job_accepted_response_serializes() {
    let resp = JobAcceptedResponse {
        job_id: "job-123".into(),
        status: JobStatus::Queued,
        job_type: JobType::Sync,
        created_at: chrono::Utc::now(),
        scheduled_for: None,
        poll_url: "/jobs/job-123".into(),
    };
    let v: Value = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["job_id"], "job-123");
    assert_eq!(v["poll_url"], "/jobs/job-123");
}

// --- JobStatusResponse ---

#[test]
fn job_status_response_serializes_complete() {
    let resp = JobStatusResponse {
        job_id: "job-456".into(),
        job_type: JobType::Index,
        status: JobStatus::Running,
        progress: 0.5,
        message: Some("halfway there".into()),
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
    assert_eq!(v["message"], "halfway there");
    assert!(v["finished_at"].is_null());
}

// --- SearchRequest ---

#[test]
fn search_request_minimal() {
    let json = json!({ "query": "rust lifetimes" });
    let req: SearchRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.query, "rust lifetimes");
    assert_eq!(req.limit, 20);
    assert_eq!(req.semantic_weight, 1.0);
    assert_eq!(req.fts_weight, 1.0);
    assert!(req.deck_names.is_none());
    assert!(req.tags.is_none());
}

#[test]
fn search_request_with_filters() {
    let json = json!({
        "query": "ownership",
        "deck_names": ["Rust"],
        "tags": ["intermediate"],
        "min_ivl": 21,
        "max_lapses": 5,
        "limit": 10,
        "semantic_weight": 0.8,
        "fts_weight": 0.2,
    });
    let req: SearchRequest = serde_json::from_value(json).unwrap();
    assert_eq!(req.deck_names.unwrap(), vec!["Rust"]);
    assert_eq!(req.limit, 10);
    assert_eq!(req.semantic_weight, 0.8);
    assert_eq!(req.min_ivl, Some(21));
}

// --- SearchResponse ---

#[test]
fn search_response_serializes() {
    let resp = SearchResponse {
        query: "test".into(),
        results: vec![SearchResultItem {
            note_id: 1,
            rrf_score: 0.95,
            semantic_score: Some(0.9),
            semantic_rank: Some(1),
            fts_score: Some(0.8),
            fts_rank: Some(2),
            headline: Some("test headline".into()),
            rerank_score: None,
            rerank_rank: None,
            sources: vec!["semantic".into(), "fts".into()],
            normalized_text: Some("normalized".into()),
            tags: Some(vec!["tag1".into()]),
            deck_names: Some(vec!["Default".into()]),
        }],
        stats: [("total".to_string(), 1i64)].into_iter().collect(),
        filters_applied: std::collections::HashMap::new(),
        lexical: None,
        rerank: None,
    };
    let v: Value = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["results"][0]["note_id"], 1);
    assert_eq!(v["results"][0]["sources"][0], "semantic");
    assert_eq!(v["stats"]["total"], 1);
}

// --- TopicCoverageResponse ---

#[test]
fn topic_coverage_response_serializes() {
    let resp = TopicCoverageResponse {
        topic_id: 42,
        path: "cs/algorithms".into(),
        label: "Algorithms".into(),
        note_count: 50,
        subtree_count: 120,
        child_count: 5,
        covered_children: 4,
        mature_count: 30,
        avg_confidence: 0.75,
        weak_notes: 10,
        avg_lapses: 1.2,
    };
    let v: Value = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["topic_id"], 42);
    assert_eq!(v["path"], "cs/algorithms");
    assert_eq!(v["avg_confidence"], 0.75);
}

// --- TopicGapsResponse ---

#[test]
fn topic_gaps_response_serializes() {
    let resp = TopicGapsResponse {
        root_path: "cs".into(),
        min_coverage: 5,
        gaps: vec![TopicGapItem {
            topic_id: 10,
            path: "cs/networking".into(),
            label: "Networking".into(),
            description: None,
            gap_type: "missing".into(),
            note_count: 0,
            threshold: 5,
        }],
        missing_count: 1,
        undercovered_count: 0,
    };
    let v: Value = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["gaps"][0]["gap_type"], "missing");
    assert_eq!(v["missing_count"], 1);
}

// --- DuplicatesResponse ---

#[test]
fn duplicates_response_serializes() {
    let resp = DuplicatesResponse {
        clusters: vec![DuplicateClusterItem {
            representative_id: 100,
            representative_text: "What is X?".into(),
            deck_names: vec!["Deck1".into()],
            tags: vec!["tag1".into()],
            duplicates: vec![DuplicateNoteItem {
                note_id: 101,
                similarity: 0.95,
                text: "What is X?".into(),
                deck_names: vec!["Deck2".into()],
                tags: vec!["tag2".into()],
            }],
            size: 2,
        }],
        stats: [("total_clusters".to_string(), json!(1))]
            .into_iter()
            .collect(),
    };
    let v: Value = serde_json::to_value(&resp).unwrap();
    assert_eq!(v["clusters"][0]["size"], 2);
    assert_eq!(v["clusters"][0]["duplicates"][0]["similarity"], 0.95);
}

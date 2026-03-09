use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// --- Sync ---

#[derive(Debug, Deserialize)]
pub struct SyncRequest {
    pub source: String,
    #[serde(default = "default_true")]
    pub run_migrations: bool,
    #[serde(default = "default_true")]
    pub index: bool,
    #[serde(default)]
    pub force_reindex: bool,
}

#[derive(Debug, Serialize)]
pub struct SyncResponse {
    pub status: String,
    pub decks_upserted: i64,
    pub models_upserted: i64,
    pub notes_upserted: i64,
    pub notes_deleted: i64,
    pub cards_upserted: i64,
    pub card_stats_upserted: i64,
    pub duration_ms: i64,
    pub notes_embedded: Option<i64>,
    pub notes_skipped: Option<i64>,
    pub index_errors: Option<Vec<String>>,
}

// --- Index ---

#[derive(Debug, Deserialize)]
pub struct IndexRequest {
    #[serde(default)]
    pub force_reindex: bool,
}

#[derive(Debug, Serialize)]
pub struct IndexResponse {
    pub status: String,
    pub notes_processed: i64,
    pub notes_embedded: i64,
    pub notes_skipped: i64,
    pub notes_deleted: i64,
    pub errors: Vec<String>,
}

// --- Async Jobs ---

#[derive(Debug, Deserialize)]
pub struct AsyncSyncRequest {
    pub source: String,
    #[serde(default = "default_true")]
    pub run_migrations: bool,
    #[serde(default = "default_true")]
    pub index: bool,
    #[serde(default)]
    pub force_reindex: bool,
    pub run_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize)]
pub struct AsyncIndexRequest {
    #[serde(default)]
    pub force_reindex: bool,
    pub run_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Serialize)]
pub struct JobAcceptedResponse {
    pub job_id: String,
    pub status: String,
    pub job_type: String,
    pub created_at: DateTime<Utc>,
    pub scheduled_for: Option<DateTime<Utc>>,
    pub poll_url: String,
}

#[derive(Debug, Serialize)]
pub struct JobStatusResponse {
    pub job_id: String,
    pub job_type: String,
    pub status: String,
    pub progress: f64,
    pub message: Option<String>,
    pub attempts: u32,
    pub max_retries: u32,
    pub cancel_requested: bool,
    pub created_at: Option<DateTime<Utc>>,
    pub scheduled_for: Option<DateTime<Utc>>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub result: Option<HashMap<String, serde_json::Value>>,
    pub error: Option<String>,
}

// --- Search ---

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub deck_names: Option<Vec<String>>,
    pub deck_names_exclude: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub tags_exclude: Option<Vec<String>>,
    pub model_ids: Option<Vec<i64>>,
    pub min_ivl: Option<i32>,
    pub max_lapses: Option<i32>,
    pub min_reps: Option<i32>,
    #[serde(default = "default_20")]
    pub limit: usize,
    #[serde(default = "default_one")]
    pub semantic_weight: f64,
    #[serde(default = "default_one")]
    pub fts_weight: f64,
}

#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    pub note_id: i64,
    pub rrf_score: f64,
    pub semantic_score: Option<f64>,
    pub semantic_rank: Option<i32>,
    pub fts_score: Option<f64>,
    pub fts_rank: Option<i32>,
    pub headline: Option<String>,
    pub rerank_score: Option<f64>,
    pub rerank_rank: Option<i32>,
    pub sources: Vec<String>,
    pub normalized_text: Option<String>,
    pub tags: Option<Vec<String>>,
    pub deck_names: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub query: String,
    pub results: Vec<SearchResultItem>,
    pub stats: HashMap<String, i64>,
    pub filters_applied: HashMap<String, serde_json::Value>,
    pub lexical: Option<HashMap<String, serde_json::Value>>,
    pub rerank: Option<HashMap<String, serde_json::Value>>,
}

// --- Topics/Coverage/Gaps ---

#[derive(Debug, Serialize)]
pub struct TopicCoverageResponse {
    pub topic_id: i64,
    pub path: String,
    pub label: String,
    pub note_count: i64,
    pub subtree_count: i64,
    pub child_count: i64,
    pub covered_children: i64,
    pub mature_count: i64,
    pub avg_confidence: f64,
    pub weak_notes: i64,
    pub avg_lapses: f64,
}

#[derive(Debug, Serialize)]
pub struct TopicGapItem {
    pub topic_id: i64,
    pub path: String,
    pub label: String,
    pub description: Option<String>,
    pub gap_type: String,
    pub note_count: i64,
    pub threshold: i64,
}

#[derive(Debug, Serialize)]
pub struct TopicGapsResponse {
    pub root_path: String,
    pub min_coverage: i64,
    pub gaps: Vec<TopicGapItem>,
    pub missing_count: i64,
    pub undercovered_count: i64,
}

// --- Duplicates ---

#[derive(Debug, Serialize)]
pub struct DuplicateNoteItem {
    pub note_id: i64,
    pub similarity: f64,
    pub text: String,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DuplicateClusterItem {
    pub representative_id: i64,
    pub representative_text: String,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
    pub duplicates: Vec<DuplicateNoteItem>,
    pub size: usize,
}

#[derive(Debug, Serialize)]
pub struct DuplicatesResponse {
    pub clusters: Vec<DuplicateClusterItem>,
    pub stats: HashMap<String, serde_json::Value>,
}

fn default_true() -> bool {
    true
}
fn default_20() -> usize {
    20
}
fn default_one() -> f64 {
    1.0
}

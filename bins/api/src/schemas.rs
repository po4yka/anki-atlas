use chrono::{DateTime, Utc};
use jobs::types::{IndexJobPayload, JobResultData, JobStatus, JobType, SyncJobPayload};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use surface_contracts::analytics::{
    DuplicateCluster, DuplicateDetail, DuplicateStats, GapKind, TopicCoverage, TopicGap, WeakNote,
};
pub use surface_contracts::search::{
    ChunkSearchHit as ChunkSearchHitItem, ChunkSearchRequest, ChunkSearchResponse, FusionStats,
    LexicalMode, SearchFilterInput as SearchFiltersDto, SearchMode, SearchRequest, SearchResponse,
    SearchResultItem,
};

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ReadyResponse {
    pub status: &'static str,
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

impl AsyncSyncRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.source.trim().is_empty() {
            return Err("source must not be empty".to_string());
        }
        Ok(())
    }
}

impl From<AsyncSyncRequest> for SyncJobPayload {
    fn from(request: AsyncSyncRequest) -> Self {
        Self {
            source: request.source,
            run_migrations: request.run_migrations,
            index: request.index,
            force_reindex: request.force_reindex,
        }
    }
}

impl From<AsyncIndexRequest> for IndexJobPayload {
    fn from(request: AsyncIndexRequest) -> Self {
        Self {
            force_reindex: request.force_reindex,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct JobAcceptedResponse {
    pub job_id: String,
    pub status: JobStatus,
    pub job_type: JobType,
    pub created_at: DateTime<Utc>,
    pub scheduled_for: Option<DateTime<Utc>>,
    pub poll_url: String,
}

#[derive(Debug, Serialize)]
pub struct JobStatusResponse {
    pub job_id: String,
    pub job_type: JobType,
    pub status: JobStatus,
    pub progress: f64,
    pub message: Option<String>,
    pub attempts: u32,
    pub max_retries: u32,
    pub cancel_requested: bool,
    pub created_at: Option<DateTime<Utc>>,
    pub scheduled_for: Option<DateTime<Utc>>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub result: Option<JobResultData>,
    pub error: Option<String>,
}

impl From<jobs::types::JobRecord> for JobStatusResponse {
    fn from(rec: jobs::types::JobRecord) -> Self {
        Self {
            job_id: rec.job_id,
            job_type: rec.job_type,
            status: rec.status,
            progress: rec.progress,
            message: rec.message,
            attempts: rec.attempts,
            max_retries: rec.max_retries,
            cancel_requested: rec.cancel_requested,
            created_at: rec.created_at,
            scheduled_for: rec.scheduled_for,
            started_at: rec.started_at,
            finished_at: rec.finished_at,
            result: rec.result,
            error: rec.error,
        }
    }
}

// --- Search ---

// --- Topics and analytics ---

#[derive(Debug, Deserialize)]
pub struct TopicsTreeQuery {
    pub root_path: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TopicsTreeResponse {
    pub topics: Vec<Value>,
}

#[derive(Debug, Deserialize)]
pub struct TopicCoverageQuery {
    pub topic_path: String,
    #[serde(default = "default_true")]
    pub include_subtree: bool,
}

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

impl From<TopicCoverage> for TopicCoverageResponse {
    fn from(coverage: TopicCoverage) -> Self {
        Self {
            topic_id: coverage.topic_id,
            path: coverage.path,
            label: coverage.label,
            note_count: coverage.note_count,
            subtree_count: coverage.subtree_count,
            child_count: coverage.child_count,
            covered_children: coverage.covered_children,
            mature_count: coverage.mature_count,
            avg_confidence: coverage.avg_confidence,
            weak_notes: coverage.weak_notes,
            avg_lapses: coverage.avg_lapses,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct TopicGapsQuery {
    pub topic_path: String,
    #[serde(default = "default_min_coverage")]
    pub min_coverage: i64,
}

impl TopicGapsQuery {
    pub fn validate(&self) -> Result<(), String> {
        if self.min_coverage <= 0 {
            return Err("min_coverage must be greater than 0".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct TopicGapItem {
    pub topic_id: i64,
    pub path: String,
    pub label: String,
    pub description: Option<String>,
    pub gap_type: GapKind,
    pub note_count: i64,
    pub threshold: i64,
    pub nearest_notes: Vec<Value>,
}

impl From<TopicGap> for TopicGapItem {
    fn from(gap: TopicGap) -> Self {
        Self {
            topic_id: gap.topic_id,
            path: gap.path,
            label: gap.label,
            description: gap.description,
            gap_type: gap.gap_type,
            note_count: gap.note_count,
            threshold: gap.threshold,
            nearest_notes: gap.nearest_notes,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct TopicGapsResponse {
    pub root_path: String,
    pub min_coverage: i64,
    pub gaps: Vec<TopicGapItem>,
    pub missing_count: usize,
    pub undercovered_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct TopicWeakNotesQuery {
    pub topic_path: String,
    #[serde(default = "default_max_results")]
    pub max_results: i64,
}

impl TopicWeakNotesQuery {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_results <= 0 {
            return Err("max_results must be greater than 0".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct TopicWeakNoteItem {
    pub note_id: common::types::NoteId,
    pub topic_path: String,
    pub confidence: f64,
    pub lapses: i32,
    pub fail_rate: Option<f64>,
    pub normalized_text: String,
}

impl From<WeakNote> for TopicWeakNoteItem {
    fn from(note: WeakNote) -> Self {
        Self {
            note_id: note.note_id,
            topic_path: note.topic_path,
            confidence: note.confidence,
            lapses: note.lapses,
            fail_rate: note.fail_rate,
            normalized_text: note.normalized_text,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct TopicWeakNotesResponse {
    pub topic_path: String,
    pub max_results: i64,
    pub notes: Vec<TopicWeakNoteItem>,
}

// --- Duplicates ---

#[derive(Debug, Deserialize)]
pub struct DuplicatesQuery {
    #[serde(default = "default_threshold")]
    pub threshold: f64,
    #[serde(default = "default_max_clusters")]
    pub max_clusters: usize,
    #[serde(alias = "deck_filter[]")]
    pub deck_filter: Option<Vec<String>>,
    #[serde(alias = "tag_filter[]")]
    pub tag_filter: Option<Vec<String>>,
}

impl DuplicatesQuery {
    pub fn from_query_string(query: Option<&str>) -> Result<Self, String> {
        let mut parsed = Self {
            threshold: default_threshold(),
            max_clusters: default_max_clusters(),
            deck_filter: None,
            tag_filter: None,
        };
        let mut deck_filter = Vec::new();
        let mut tag_filter = Vec::new();

        for (key, value) in url::form_urlencoded::parse(query.unwrap_or_default().as_bytes()) {
            match key.as_ref() {
                "threshold" => {
                    parsed.threshold = value
                        .parse()
                        .map_err(|_| "threshold must be a floating point number".to_string())?;
                }
                "max_clusters" => {
                    parsed.max_clusters = value
                        .parse()
                        .map_err(|_| "max_clusters must be an integer".to_string())?;
                }
                "deck_filter" | "deck_filter[]" => deck_filter.push(value.into_owned()),
                "tag_filter" | "tag_filter[]" => tag_filter.push(value.into_owned()),
                _ => {}
            }
        }

        if !deck_filter.is_empty() {
            parsed.deck_filter = Some(deck_filter);
        }
        if !tag_filter.is_empty() {
            parsed.tag_filter = Some(tag_filter);
        }

        parsed.validate()?;
        Ok(parsed)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.threshold) {
            return Err("threshold must be between 0.0 and 1.0".to_string());
        }
        if self.max_clusters == 0 {
            return Err("max_clusters must be greater than 0".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct DuplicateNoteItem {
    pub note_id: common::types::NoteId,
    pub similarity: f64,
    pub text: String,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
}

impl From<DuplicateDetail> for DuplicateNoteItem {
    fn from(note: DuplicateDetail) -> Self {
        Self {
            note_id: note.note_id,
            similarity: note.similarity,
            text: note.text,
            deck_names: note.deck_names,
            tags: note.tags,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct DuplicateClusterItem {
    pub representative_id: common::types::NoteId,
    pub representative_text: String,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
    pub duplicates: Vec<DuplicateNoteItem>,
    pub size: usize,
}

impl From<DuplicateCluster> for DuplicateClusterItem {
    fn from(cluster: DuplicateCluster) -> Self {
        let size = cluster.size();

        Self {
            representative_id: cluster.representative_id,
            representative_text: cluster.representative_text,
            deck_names: cluster.deck_names,
            tags: cluster.tags,
            size,
            duplicates: cluster.duplicates.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct DuplicateStatsResponse {
    pub notes_scanned: usize,
    pub clusters_found: usize,
    pub total_duplicates: usize,
    pub avg_cluster_size: f64,
}

impl From<DuplicateStats> for DuplicateStatsResponse {
    fn from(stats: DuplicateStats) -> Self {
        Self {
            notes_scanned: stats.notes_scanned,
            clusters_found: stats.clusters_found,
            total_duplicates: stats.total_duplicates,
            avg_cluster_size: stats.avg_cluster_size,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct DuplicatesResponse {
    pub clusters: Vec<DuplicateClusterItem>,
    pub stats: DuplicateStatsResponse,
}

fn default_true() -> bool {
    true
}

fn default_min_coverage() -> i64 {
    1
}

fn default_max_results() -> i64 {
    20
}

fn default_threshold() -> f64 {
    0.92
}

fn default_max_clusters() -> usize {
    50
}

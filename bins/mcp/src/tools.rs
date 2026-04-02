use rmcp::schemars;
use rmcp::schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

fn default_true() -> bool {
    true
}

fn default_limit() -> usize {
    10
}

fn default_max_results() -> i64 {
    20
}

fn default_min_coverage() -> i64 {
    1
}

fn default_threshold() -> f64 {
    0.92
}

fn default_max_clusters() -> usize {
    50
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OutputMode {
    #[default]
    Markdown,
    Json,
}

/// Search mode for controlling which retrieval sources are used.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum McpSearchMode {
    /// Use both semantic and FTS sources with RRF fusion (default).
    #[default]
    Hybrid,
    /// Use semantic (vector) search only.
    SemanticOnly,
    /// Use full-text search only.
    FtsOnly,
}

impl From<McpSearchMode> for surface_contracts::search::SearchMode {
    fn from(mode: McpSearchMode) -> Self {
        match mode {
            McpSearchMode::Hybrid => surface_contracts::search::SearchMode::Hybrid,
            McpSearchMode::SemanticOnly => surface_contracts::search::SearchMode::SemanticOnly,
            McpSearchMode::FtsOnly => surface_contracts::search::SearchMode::FtsOnly,
        }
    }
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SearchToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    pub query: String,
    #[serde(default)]
    pub deck_names: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub search_mode: McpSearchMode,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ChunkSearchToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    pub query: String,
    #[serde(default)]
    pub deck_names: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct TopicsToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    pub root_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct TopicCoverageToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    pub topic_path: String,
    #[serde(default = "default_true")]
    pub include_subtree: bool,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct TopicGapsToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    pub topic_path: String,
    #[serde(default = "default_min_coverage")]
    pub min_coverage: i64,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct TopicWeakNotesToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    pub topic_path: String,
    #[serde(default = "default_max_results")]
    pub max_results: i64,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct DuplicatesToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    #[serde(default = "default_threshold")]
    pub threshold: f64,
    #[serde(default = "default_max_clusters")]
    pub max_clusters: usize,
    #[serde(default)]
    pub deck_filter: Vec<String>,
    #[serde(default)]
    pub tag_filter: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SyncJobToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    pub source: String,
    #[serde(default = "default_true")]
    pub run_migrations: bool,
    #[serde(default = "default_true")]
    pub index: bool,
    #[serde(default)]
    pub force_reindex: bool,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct IndexJobToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    #[serde(default)]
    pub force_reindex: bool,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct JobStatusToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    pub job_id: String,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct JobCancelToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    pub job_id: String,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct GenerateToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    pub file_path: String,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ValidateToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    pub file_path: String,
    #[serde(default)]
    pub quality: bool,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ObsidianSyncToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    pub vault_path: String,
    #[serde(default)]
    pub source_dirs: Vec<String>,
    #[serde(default = "default_true")]
    pub dry_run: bool,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct TagAuditToolInput {
    #[serde(default)]
    pub output_mode: OutputMode,
    pub file_path: String,
    #[serde(default)]
    pub fix: bool,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SearchResultView {
    pub note_id: i64,
    pub rrf_score: f64,
    pub semantic_score: Option<f64>,
    pub fts_score: Option<f64>,
    pub rerank_score: Option<f64>,
    pub headline: Option<String>,
    pub sources: Vec<String>,
    pub match_modality: Option<String>,
    pub match_chunk_kind: Option<String>,
    pub match_source_field: Option<String>,
    pub match_asset_rel_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct SearchToolResult {
    pub query: String,
    pub total_results: usize,
    pub lexical_mode: String,
    pub lexical_fallback_used: bool,
    pub rerank_applied: bool,
    pub query_suggestions: Vec<String>,
    pub autocomplete_suggestions: Vec<String>,
    pub results: Vec<SearchResultView>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ChunkSearchResultView {
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

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ChunkSearchToolResult {
    pub query: String,
    pub total_results: usize,
    pub results: Vec<ChunkSearchResultView>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct TopicsToolResult {
    pub root_path: Option<String>,
    pub topic_count: usize,
    pub topics: Value,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct TopicCoverageToolResult {
    pub topic_path: String,
    pub found: bool,
    pub coverage: Option<Value>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct TopicGapsToolResult {
    pub topic_path: String,
    pub min_coverage: i64,
    pub gaps: Value,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct TopicWeakNotesToolResult {
    pub topic_path: String,
    pub max_results: i64,
    pub notes: Value,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct DuplicatesToolResult {
    pub threshold: f64,
    pub max_clusters: usize,
    pub clusters: Value,
    pub stats: Value,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct JobAcceptedToolResult {
    pub job_id: String,
    pub job_type: String,
    pub status: String,
    pub poll_hint: String,
    pub cancel_hint: String,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct JobStatusToolResult {
    pub job_id: String,
    pub job_type: String,
    pub status: String,
    pub progress: f64,
    pub message: Option<String>,
    pub result: Option<Value>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct WorkflowToolResult {
    pub path: String,
    pub summary: String,
    pub data: Value,
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct ToolError {
    pub error: String,
    pub message: String,
    pub details: Option<String>,
}

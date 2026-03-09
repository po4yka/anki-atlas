use serde::Deserialize;

/// Default timeout for MCP tool operations.
pub const TOOL_TIMEOUT_SECS: u64 = 30;

/// Sync operation timeout (large collections).
pub const SYNC_TIMEOUT_SECS: u64 = 120;

/// Index operation timeout.
pub const INDEX_TIMEOUT_SECS: u64 = 300;

// --- Serde default helpers ---

fn default_search_limit() -> usize {
    20
}

fn default_true() -> bool {
    true
}

fn default_min_coverage() -> usize {
    1
}

fn default_threshold() -> f64 {
    0.92
}

fn default_max_clusters() -> usize {
    50
}

// --- Tool Input Types ---

#[derive(Debug, Deserialize)]
pub struct SearchInput {
    pub query: String,
    #[serde(default = "default_search_limit")]
    pub limit: usize,
    pub deck_filter: Option<Vec<String>>,
    pub tag_filter: Option<Vec<String>>,
    #[serde(default)]
    pub semantic_only: bool,
    #[serde(default)]
    pub fts_only: bool,
}

#[derive(Debug, Deserialize)]
pub struct TopicCoverageInput {
    pub topic_path: String,
    #[serde(default = "default_true")]
    pub include_subtree: bool,
}

#[derive(Debug, Deserialize)]
pub struct TopicGapsInput {
    pub topic_path: String,
    #[serde(default = "default_min_coverage")]
    pub min_coverage: usize,
}

#[derive(Debug, Deserialize)]
pub struct DuplicatesInput {
    #[serde(default = "default_threshold")]
    pub threshold: f64,
    #[serde(default = "default_max_clusters")]
    pub max_clusters: usize,
    pub deck_filter: Option<Vec<String>>,
    pub tag_filter: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct SyncInput {
    pub collection_path: String,
    #[serde(default)]
    pub run_index: bool,
}

#[derive(Debug, Deserialize)]
pub struct GenerateInput {
    pub text: String,
    pub deck: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ValidateInput {
    pub front: String,
    pub back: String,
    pub tags: Option<Vec<String>>,
    #[serde(default = "default_true")]
    pub check_quality: bool,
}

#[derive(Debug, Deserialize)]
pub struct ObsidianSyncInput {
    pub vault_path: String,
    #[serde(default = "default_true")]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
pub struct TagAuditInput {
    pub tags: Vec<String>,
    #[serde(default)]
    pub fix: bool,
}

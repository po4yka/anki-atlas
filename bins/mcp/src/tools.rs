use serde::Deserialize;

/// Default timeout for MCP tool operations.
pub const TOOL_TIMEOUT_SECS: u64 = 30;

// --- Serde default helpers ---

fn default_true() -> bool {
    true
}

// --- Tool Input Types ---

#[derive(Debug, Deserialize)]
pub struct GenerateInput {
    pub text: String,
    pub deck: Option<String>,
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

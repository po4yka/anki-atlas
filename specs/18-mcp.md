# Spec: crate `mcp`

## Source Reference
Python: `apps/mcp/` (tools.py, formatters.py, server.py, instance.py, cli.py)

## Purpose
Model Context Protocol (MCP) server exposing anki-atlas tools for AI agent consumption. Provides 8 tools: search, topic coverage, topic gaps, duplicates, sync, generate preview, validate, obsidian sync, and tag audit. Each tool returns markdown-formatted results. Uses `rmcp` crate for MCP protocol handling over stdio transport. Includes timeout handling (30s default, 120s for sync) and structured error messages with actionable guidance.

## Dependencies
```toml
[dependencies]
common = { path = "../common" }
obsidian = { path = "../obsidian" }
# Additional workspace crates: search, analytics, anki-sync, indexer, validation, taxonomy, generator

rmcp = { version = "0.1", features = ["server", "transport-io"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["rt-multi-thread", "macros", "time", "signal"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
anyhow = "1"

[dev-dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Public API

### Tool Definitions (`src/tools.rs`)

Each tool is registered with the rmcp server and has typed input parameters.

```rust
use serde::{Deserialize, Serialize};

/// Default timeout for MCP tool operations.
pub const TOOL_TIMEOUT_SECS: u64 = 30;

/// Sync operation timeout (large collections).
pub const SYNC_TIMEOUT_SECS: u64 = 120;

/// Index operation timeout.
pub const INDEX_TIMEOUT_SECS: u64 = 300;

// --- Tool Input Types ---

#[derive(Debug, Deserialize)]
pub struct SearchInput {
    pub query: String,
    #[serde(default = "default_20")]
    pub limit: usize,              // 1-100
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
    #[serde(default = "default_1")]
    pub min_coverage: usize,
}

#[derive(Debug, Deserialize)]
pub struct DuplicatesInput {
    #[serde(default = "default_threshold")]
    pub threshold: f64,            // 0.5-1.0
    #[serde(default = "default_50")]
    pub max_clusters: usize,       // 1-500
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

fn default_20() -> usize { 20 }
fn default_1() -> usize { 1 }
fn default_50() -> usize { 50 }
fn default_true() -> bool { true }
fn default_threshold() -> f64 { 0.92 }
```

### Tool Handler Functions (`src/handlers.rs`)

```rust
/// Each handler:
/// 1. Deserializes input
/// 2. Wraps service call in tokio::time::timeout
/// 3. Formats result as markdown via formatters
/// 4. Returns formatted string or error guidance

pub async fn handle_search(input: SearchInput) -> String;
pub async fn handle_topic_coverage(input: TopicCoverageInput) -> String;
pub async fn handle_topic_gaps(input: TopicGapsInput) -> String;
pub async fn handle_duplicates(input: DuplicatesInput) -> String;
pub async fn handle_sync(input: SyncInput) -> String;
pub async fn handle_generate(input: GenerateInput) -> String;
pub async fn handle_validate(input: ValidateInput) -> String;
pub async fn handle_obsidian_sync(input: ObsidianSyncInput) -> String;
pub async fn handle_tag_audit(input: TagAuditInput) -> String;
```

### Formatters (`src/formatters.rs`)

```rust
/// Format search results as markdown table.
pub fn format_search_result(/* ... */) -> String;

/// Format topic coverage as markdown with metrics sections.
pub fn format_coverage_result(/* ... */) -> String;

/// Format topic gaps as markdown with missing/undercovered tables.
pub fn format_gaps_result(/* ... */) -> String;

/// Format duplicate clusters as markdown.
pub fn format_duplicates_result(/* ... */) -> String;

/// Format sync stats as markdown.
pub fn format_sync_result(/* ... */) -> String;

/// Format note parse result for generation preview.
pub fn format_generate_result(
    title: Option<&str>,
    sections: &[(String, String)],
    body_length: usize,
) -> String;

/// Format validation result with optional quality scores.
pub fn format_validate_result(/* ... */) -> String;

/// Format obsidian vault scan result.
pub fn format_obsidian_sync_result(
    notes_found: usize,
    parsed_notes: &[(String, Option<String>, usize)],
    vault_path: &str,
) -> String;

/// Format tag audit result.
pub fn format_tag_audit_result(
    results: &[(String, Vec<String>, Option<String>, Vec<String>)],
) -> String;

/// Truncate text with ellipsis at max_len.
fn truncate(text: &str, max_len: usize) -> String;
```

### Server Setup (`src/server.rs`)

```rust
/// Configure and run the MCP server over stdio transport.
///
/// Registers all tools with the rmcp server, configures logging
/// to stderr (to avoid polluting stdio MCP protocol), and runs
/// until stdin closes or SIGTERM is received.
pub async fn run_server() -> anyhow::Result<()>;
```

### Entry Point (`src/main.rs`)

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    server::run_server().await
}
```

### Module structure

```
src/
  main.rs          -- Entry point
  server.rs        -- MCP server setup, tool registration
  tools.rs         -- Tool input types, constants
  handlers.rs      -- Tool handler implementations
  formatters.rs    -- Markdown output formatting
```

## Internal Details

### Error Formatting
Each tool catches all errors and returns a structured markdown error message:
- **Database unavailable**: suggests checking PostgreSQL, running migrations, checking env vars.
- **Vector store unavailable**: suggests checking Qdrant, checking env vars, running sync.
- **Timeout**: suggests more specific query, reducing limit, checking service load.
- **Other**: shows error type and message.

### Sync Tool
- Validates `.anki2` extension and path existence.
- Runs sync with 120s timeout.
- Optionally runs indexing with 300s timeout.
- Returns combined markdown result.

### Generate Tool
- Writes input text to a temp file.
- Parses with `obsidian::parse_note`.
- Returns generation preview (title, sections, estimated cards).
- Cleans up temp file.

### Validate Tool
- Runs `ValidationPipeline` with ContentValidator, FormatValidator, HTMLValidator, TagValidator.
- Optionally runs quality assessment.
- Formats PASS/FAIL with error/warning counts and quality scores.

### Obsidian Sync Tool
- Validates vault path is a directory.
- Discovers notes, parses up to 50 for performance.
- Returns scan summary with note/section counts.

### Tag Audit Tool
- For each tag: validate, optionally normalize, suggest close matches.
- Returns summary with valid/invalid counts and issues table.

### Logging
- All logging goes to stderr (not stdout) to avoid interfering with MCP stdio protocol.
- Uses `tracing-subscriber` with JSON format and env-filter.

## Acceptance Criteria
- [ ] MCP server starts and registers all 9 tools (search, coverage, gaps, duplicates, sync, generate, validate, obsidian-sync, tag-audit)
- [ ] `ankiatlas_search` accepts query with filters and returns markdown table
- [ ] `ankiatlas_search` enforces limit 1-100
- [ ] `ankiatlas_topic_coverage` returns coverage metrics or "not found" message
- [ ] `ankiatlas_topic_gaps` returns missing/undercovered tables
- [ ] `ankiatlas_duplicates` respects threshold 0.5-1.0 and max_clusters 1-500
- [ ] `ankiatlas_sync` validates .anki2 path and returns sync stats
- [ ] `ankiatlas_sync` optionally runs indexing after sync
- [ ] `ankiatlas_generate` parses markdown text and returns preview
- [ ] `ankiatlas_validate` runs validation pipeline and returns PASS/FAIL
- [ ] `ankiatlas_obsidian_sync` discovers vault notes and returns scan summary
- [ ] `ankiatlas_tag_audit` validates tags and shows issues
- [ ] Timeout errors return actionable guidance messages
- [ ] Database/vector store errors return service-specific guidance
- [ ] All formatters produce valid markdown output
- [ ] `truncate` correctly handles text shorter and longer than max_len
- [ ] Logging goes to stderr, not stdout
- [ ] Server shuts down cleanly on stdin close or SIGTERM
- [ ] `make check` equivalent passes (clippy, fmt, test)

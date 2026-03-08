# Spec: crate `obsidian`

## Source Reference
Python: `packages/obsidian/` (parser.py, analyzer.py, frontmatter.py, sync.py)

## Purpose
Parse Obsidian vault markdown notes into structured representations. Handles frontmatter extraction (YAML), section splitting by headings, title detection, vault discovery with ignore patterns, wikilink extraction, and vault-level statistics (orphaned notes, broken links). The sync workflow orchestrates discovery, card generation, validation, and Anki sync. All operations are synchronous filesystem reads wrapped in `tokio::task::spawn_blocking` where needed.

## Dependencies
```toml
[dependencies]
common = { path = "../common" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
serde_yaml = "0.9"
thiserror = "2"
tracing = "0.1"
regex = "1"
walkdir = "2"

[dev-dependencies]
tempfile = "3"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Public API

### Error (`src/error.rs`)

```rust
use thiserror::Error;
use std::path::PathBuf;

#[derive(Debug, Error)]
pub enum ObsidianError {
    #[error("parse error: {message} (path: {path:?})")]
    Parse { message: String, path: Option<PathBuf> },

    #[error("file not found: {0}")]
    NotFound(PathBuf),

    #[error("file too large: {path} ({size} bytes, max {max} bytes)")]
    FileTooLarge { path: PathBuf, size: u64, max: u64 },

    #[error("path outside vault root: {path} (root: {root})")]
    OutsideVault { path: PathBuf, root: PathBuf },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("YAML error: {0}")]
    Yaml(String),
}
```

### Frontmatter (`src/frontmatter.rs`)

```rust
use std::collections::HashMap;
use crate::error::ObsidianError;

/// Extract YAML frontmatter from note content.
/// Returns empty map if no frontmatter block is found.
/// Preprocesses YAML to fix common syntax errors (backticks in values).
pub fn parse_frontmatter(content: &str) -> Result<HashMap<String, serde_yaml::Value>, ObsidianError>;

/// Write or replace YAML frontmatter in note content.
/// Returns the full content with updated frontmatter block.
pub fn write_frontmatter(
    data: &HashMap<String, serde_yaml::Value>,
    content: &str,
) -> Result<String, ObsidianError>;
```

### ParsedNote (`src/parser.rs`)

```rust
use std::path::PathBuf;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Maximum file size: 10 MB.
pub const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024;

/// Default directories to ignore during discovery.
pub const DEFAULT_IGNORE_DIRS: &[&str] = &[".obsidian", ".trash", ".git"];

/// A parsed Obsidian markdown note.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedNote {
    pub path: PathBuf,
    pub frontmatter: HashMap<String, serde_yaml::Value>,
    pub content: String,
    pub body: String,
    pub sections: Vec<(String, String)>,  // (heading, content) pairs
    pub title: Option<String>,
}

/// Parse a single markdown note from disk.
///
/// Validates: file exists, is a file, within vault root (if given), under MAX_FILE_SIZE.
/// Extracts frontmatter, body (content after frontmatter), sections, and title.
pub fn parse_note(
    path: &std::path::Path,
    vault_root: Option<&std::path::Path>,
) -> Result<ParsedNote, ObsidianError>;

/// Discover all markdown notes in a vault directory.
///
/// Recursively walks `vault_root`, matching `patterns` (default: `*.md`),
/// skipping directories in `ignore_dirs`. Validates symlinks stay within vault.
/// Returns sorted list of absolute paths.
pub fn discover_notes(
    vault_root: &std::path::Path,
    patterns: &[&str],       // default &["*.md"]
    ignore_dirs: &[&str],    // default DEFAULT_IGNORE_DIRS
) -> Result<Vec<PathBuf>, ObsidianError>;
```

### Analyzer (`src/analyzer.rs`)

```rust
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

/// Statistics about an Obsidian vault.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultStats {
    pub total_notes: usize,
    pub total_dirs: usize,
    pub notes_with_frontmatter: usize,
    pub wikilinks_count: usize,
    pub orphaned_notes: Vec<String>,
    pub broken_links: Vec<(String, String)>,  // (source_stem, target)
}

/// Analyze vault structure: wikilinks, orphans, broken links.
pub struct VaultAnalyzer {
    vault_path: PathBuf,
    // internal state...
}

impl VaultAnalyzer {
    pub fn new(vault_path: &Path) -> Self;

    /// Get wikilink targets from a specific note.
    pub fn get_wikilinks(&mut self, path: &Path) -> Vec<String>;

    /// Find notes with no incoming or outgoing links.
    pub fn find_orphaned(&mut self) -> Vec<PathBuf>;

    /// Compute comprehensive vault statistics.
    pub fn analyze(&mut self) -> VaultStats;
}
```

### Sync Workflow (`src/sync.rs`)

```rust
use std::path::{Path, PathBuf};
use crate::parser::ParsedNote;

/// Progress callback: (phase, current, total).
pub type ProgressCallback = Box<dyn Fn(&str, usize, usize) + Send + Sync>;

/// Result of processing a single note.
#[derive(Debug, Clone)]
pub struct NoteResult {
    pub note_path: PathBuf,
    pub cards_generated: usize,
    pub errors: Vec<String>,
}

/// Workflow-level sync result with counts.
#[derive(Debug, Clone, Default)]
pub struct SyncResult {
    pub generated: usize,
    pub updated: usize,
    pub skipped: usize,
    pub failed: usize,
    pub errors: Vec<String>,
}

impl SyncResult {
    /// Combine two results.
    pub fn merge(self, other: SyncResult) -> SyncResult;
}

/// Trait for card generation from a parsed note (injected dependency).
pub trait CardGenerator: Send + Sync {
    fn generate(&self, note: &ParsedNote) -> Vec<GeneratedCardRef>;
}

/// Placeholder for generated card reference (actual type from generator crate).
#[derive(Debug, Clone)]
pub struct GeneratedCardRef {
    pub slug: String,
    pub apf_html: String,
}

/// Orchestrates: discover notes -> generate cards -> validate -> aggregate.
pub struct ObsidianSyncWorkflow<G: CardGenerator> {
    generator: G,
    on_progress: Option<ProgressCallback>,
}

impl<G: CardGenerator> ObsidianSyncWorkflow<G> {
    pub fn new(generator: G, on_progress: Option<ProgressCallback>) -> Self;

    /// Discover and parse all notes in vault.
    pub fn scan_vault(
        &self,
        vault_path: &Path,
        source_dirs: Option<&[&str]>,
    ) -> Vec<ParsedNote>;

    /// Full pipeline: scan -> process all notes -> aggregate results.
    pub fn run(
        &self,
        vault_path: &Path,
        source_dirs: Option<&[&str]>,
    ) -> SyncResult;
}
```

### Module root (`src/lib.rs`)

```rust
pub mod analyzer;
pub mod error;
pub mod frontmatter;
pub mod parser;
pub mod sync;

pub use error::ObsidianError;
pub use parser::{ParsedNote, parse_note, discover_notes, MAX_FILE_SIZE};
pub use analyzer::{VaultAnalyzer, VaultStats};
pub use frontmatter::{parse_frontmatter, write_frontmatter};
```

## Internal Details

### Frontmatter Parsing
- Detect YAML block between `---\n...\n---\n` at start of file.
- Preprocess: strip backticks from YAML values (`\`value\`` -> `value`).
- Parse with `serde_yaml`. Return empty map on no frontmatter.
- Body is content after the closing `---\n` delimiter.

### Title Extraction
1. Check `frontmatter["title"]` first.
2. Fall back to first `# Heading` in body via regex `^#\s+(.+)$`.
3. Return `None` if neither found.

### Section Splitting
- Regex: `^(#{1,6}\s+.+)$` (multiline).
- Content before first heading gets `("", content)` entry.
- Each heading starts a section that extends to the next heading or EOF.

### Path Validation
- Resolve path, check exists, check is_file.
- If `vault_root` provided: ensure resolved path is under resolved root (traversal protection).
- Check `metadata().len()` against `MAX_FILE_SIZE`.

### Wikilink Regex
- Pattern: `\[\[([^\]|]+)(?:\|[^\]]+)?\]\]` -- captures target, ignores alias.

### VaultAnalyzer Lazy Scanning
- `_scan()` called lazily on first query. Sets `_scanned = true`.
- Builds `links: HashMap<String, Vec<String>>` mapping note stem to wikilink targets.
- Orphaned = notes with no outgoing links AND not linked to by any other note.

## Acceptance Criteria
- [ ] `parse_frontmatter` extracts YAML key-value pairs from valid frontmatter
- [ ] `parse_frontmatter` returns empty map when no frontmatter block exists
- [ ] `parse_frontmatter` handles backtick-in-value preprocessing
- [ ] `parse_frontmatter` returns error on malformed YAML
- [ ] `write_frontmatter` replaces existing frontmatter block
- [ ] `write_frontmatter` adds frontmatter to content without one
- [ ] `parse_note` extracts title from frontmatter `title` field
- [ ] `parse_note` extracts title from first `# Heading` when no frontmatter title
- [ ] `parse_note` returns `None` title when neither source exists
- [ ] `parse_note` splits body into sections by heading
- [ ] `parse_note` puts pre-heading content into `("", content)` section
- [ ] `parse_note` returns error for nonexistent file
- [ ] `parse_note` returns error for file exceeding MAX_FILE_SIZE
- [ ] `parse_note` returns error for path outside vault root
- [ ] `discover_notes` finds all `.md` files recursively
- [ ] `discover_notes` skips `.obsidian`, `.trash`, `.git` directories
- [ ] `discover_notes` returns sorted paths
- [ ] `discover_notes` skips symlinks that resolve outside vault root
- [ ] `discover_notes` returns error for nonexistent vault root
- [ ] `VaultAnalyzer::get_wikilinks` extracts `[[target]]` and `[[target|alias]]`
- [ ] `VaultAnalyzer::find_orphaned` returns notes with no incoming or outgoing links
- [ ] `VaultAnalyzer::analyze` computes correct total_notes, total_dirs, broken_links
- [ ] `SyncResult::merge` combines counts and errors from two results
- [ ] `ObsidianSyncWorkflow::scan_vault` respects source_dirs filter
- [ ] All types are `Send + Sync` (compile-time assertion)

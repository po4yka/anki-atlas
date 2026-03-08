use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::ObsidianError;

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
    pub sections: Vec<(String, String)>,
    pub title: Option<String>,
}

/// Parse a single markdown note from disk.
///
/// Validates: file exists, is a file, within vault root (if given), under MAX_FILE_SIZE.
/// Extracts frontmatter, body (content after frontmatter), sections, and title.
pub fn parse_note(
    _path: &Path,
    _vault_root: Option<&Path>,
) -> Result<ParsedNote, ObsidianError> {
    todo!()
}

/// Discover all markdown notes in a vault directory.
///
/// Recursively walks `vault_root`, matching `patterns` (default: `*.md`),
/// skipping directories in `ignore_dirs`. Validates symlinks stay within vault.
/// Returns sorted list of absolute paths.
pub fn discover_notes(
    _vault_root: &Path,
    _patterns: &[&str],
    _ignore_dirs: &[&str],
) -> Result<Vec<PathBuf>, ObsidianError> {
    todo!()
}

#[cfg(test)]
mod tests;

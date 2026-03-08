use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use regex::Regex;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use crate::error::ObsidianError;
use crate::frontmatter::parse_frontmatter;

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
pub fn parse_note(path: &Path, vault_root: Option<&Path>) -> Result<ParsedNote, ObsidianError> {
    // Validation: exists
    if !path.exists() {
        return Err(ObsidianError::NotFound(path.to_path_buf()));
    }

    let resolved = path.canonicalize().map_err(ObsidianError::Io)?;

    // Validation: vault root containment
    if let Some(root) = vault_root {
        let resolved_root = root.canonicalize().map_err(ObsidianError::Io)?;
        if !resolved.starts_with(&resolved_root) {
            return Err(ObsidianError::OutsideVault {
                path: path.to_path_buf(),
                root: root.to_path_buf(),
            });
        }
    }

    // Validation: file size
    let metadata = fs::metadata(&resolved).map_err(ObsidianError::Io)?;
    let size = metadata.len();
    if size > MAX_FILE_SIZE {
        return Err(ObsidianError::FileTooLarge {
            path: path.to_path_buf(),
            size,
            max: MAX_FILE_SIZE,
        });
    }

    let content = fs::read_to_string(&resolved).map_err(ObsidianError::Io)?;
    let frontmatter = parse_frontmatter(&content)?;
    let body = extract_body(&content);
    let title = extract_title(&frontmatter, &body);
    let sections = split_sections(&body);

    Ok(ParsedNote {
        path: path.to_path_buf(),
        frontmatter,
        content,
        body,
        sections,
        title,
    })
}

/// Discover all markdown notes in a vault directory.
///
/// Recursively walks `vault_root`, matching `patterns` (default: `*.md`),
/// skipping directories in `ignore_dirs`. Validates symlinks stay within vault.
/// Returns sorted list of absolute paths.
pub fn discover_notes(
    vault_root: &Path,
    patterns: &[&str],
    ignore_dirs: &[&str],
) -> Result<Vec<PathBuf>, ObsidianError> {
    if !vault_root.exists() {
        return Err(ObsidianError::NotFound(vault_root.to_path_buf()));
    }

    let resolved_root = vault_root.canonicalize().map_err(ObsidianError::Io)?;

    let mut paths = Vec::new();

    for entry in WalkDir::new(&resolved_root).follow_links(false) {
        let entry = entry.map_err(|e| {
            ObsidianError::Io(e.into_io_error().unwrap_or_else(|| {
                std::io::Error::other("walkdir error")
            }))
        })?;

        // Skip ignored directories
        if entry.file_type().is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                if ignore_dirs.contains(&name) {
                    continue;
                }
            }
        }

        // Check if it's inside an ignored directory by checking path components
        let rel_path = entry.path().strip_prefix(&resolved_root).unwrap_or(entry.path());
        if rel_path
            .components()
            .any(|c| ignore_dirs.contains(&c.as_os_str().to_str().unwrap_or("")))
        {
            continue;
        }

        if !entry.file_type().is_file() && !entry.file_type().is_symlink() {
            continue;
        }

        // Skip symlinks pointing outside vault
        if entry.path_is_symlink() {
            if let Ok(target) = fs::canonicalize(entry.path()) {
                if !target.starts_with(&resolved_root) {
                    continue;
                }
            } else {
                continue;
            }
        }

        let path = entry.path();

        // Match against patterns
        if !matches_any_pattern(path, patterns) {
            continue;
        }

        paths.push(path.to_path_buf());
    }

    paths.sort();
    Ok(paths)
}

/// Extract the body (content after frontmatter) from note content.
fn extract_body(content: &str) -> String {
    let Some(rest) = content.strip_prefix("---\n") else {
        return content.to_string();
    };
    let Some(end) = rest.find("\n---") else {
        return content.to_string();
    };
    let after_delim = &rest[end + 4..]; // skip "\n---"
    after_delim
        .strip_prefix('\n')
        .unwrap_or(after_delim)
        .to_string()
}

/// Extract title: frontmatter "title" first, then first H1 heading.
fn extract_title(frontmatter: &HashMap<String, serde_yaml::Value>, body: &str) -> Option<String> {
    // Check frontmatter title
    if let Some(val) = frontmatter.get("title") {
        if let Some(s) = val.as_str() {
            return Some(s.to_string());
        }
    }

    // Fall back to first H1 heading
    let re = Regex::new(r"(?m)^#\s+(.+)$").ok()?;
    re.captures(body).map(|c| c[1].to_string())
}

/// Split body into sections by headings.
fn split_sections(body: &str) -> Vec<(String, String)> {
    let re = Regex::new(r"(?m)^(#{1,6}\s+.+)$").expect("valid regex");
    let mut sections = Vec::new();

    let matches: Vec<_> = re.find_iter(body).collect();

    if matches.is_empty() {
        return vec![("".to_string(), body.to_string())];
    }

    // Pre-heading content
    let first_start = matches[0].start();
    if first_start > 0 {
        sections.push(("".to_string(), body[..first_start].to_string()));
    }

    for (i, m) in matches.iter().enumerate() {
        let heading = m.as_str().to_string();
        let content_start = m.end();
        let content_end = matches.get(i + 1).map_or(body.len(), |next| next.start());
        let content = body[content_start..content_end].to_string();
        // Strip leading newline from content
        let content = content.strip_prefix('\n').unwrap_or(&content).to_string();
        sections.push((heading, content));
    }

    sections
}

/// Check if a path matches any of the given glob patterns (simple extension matching).
fn matches_any_pattern(path: &Path, patterns: &[&str]) -> bool {
    for pattern in patterns {
        // Simple *.ext pattern matching
        if let Some(ext_pattern) = pattern.strip_prefix("*.") {
            if let Some(ext) = path.extension() {
                if ext == ext_pattern {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests;

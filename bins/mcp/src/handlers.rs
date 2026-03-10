// Tool handler implementations.

use std::path::Path;

use tracing::instrument;

use crate::formatters::{
    format_generate_result, format_obsidian_sync_result, format_tag_audit_result, TagAuditEntry,
};
use crate::tools::{GenerateInput, ObsidianSyncInput, TagAuditInput};

/// Categories of errors returned by handler operations.
#[derive(Debug)]
pub enum ErrorKind {
    DatabaseUnavailable,
    VectorStoreUnavailable,
    Timeout,
    Other { error_type: String, message: String },
}

/// Format an error into a user-friendly markdown message with actionable guidance.
pub fn format_error(kind: ErrorKind, operation: &str) -> String {
    match kind {
        ErrorKind::DatabaseUnavailable => {
            format!(
                "**Database unavailable** during `{operation}`.\n\n\
                 Check that PostgreSQL is running and accessible."
            )
        }
        ErrorKind::VectorStoreUnavailable => {
            format!(
                "**Vector store unavailable** during `{operation}`.\n\n\
                 Check that Qdrant is running and accessible."
            )
        }
        ErrorKind::Timeout => {
            format!(
                "**Operation timed out** during `{operation}`.\n\n\
                 The `{operation}` operation timed out. \
                 Try reducing the limit or using a more specific query."
            )
        }
        ErrorKind::Other {
            error_type,
            message,
        } => {
            format!(
                "**Error during `{operation}`**: {error_type}\n\n{message}"
            )
        }
    }
}

/// Clamp a value to the range [min, max].
pub fn clamp_limit(value: usize, min: usize, max: usize) -> usize {
    value.clamp(min, max)
}

/// Validate that a path points to an existing directory.
/// Returns `Ok(())` on success, `Err(message)` on failure.
pub fn validate_vault_path(path: &str) -> Result<(), String> {
    let p = Path::new(path);

    if !p.exists() {
        return Err(format!("Path not found: `{path}` does not exist"));
    }

    if !p.is_dir() {
        return Err(format!(
            "Path is not a directory: `{path}`"
        ));
    }

    Ok(())
}

/// Parse markdown text and return a generation preview.
#[instrument(skip(input), fields(text_len = input.text.len()))]
pub async fn handle_generate(input: GenerateInput) -> String {
    let text = &input.text;

    // Extract title from first H1
    let title = text
        .lines()
        .find(|line| line.starts_with("# "))
        .map(|line| line.trim_start_matches("# ").trim());

    // Extract sections from H2 headings
    let mut sections: Vec<(String, String)> = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut current_content = String::new();

    for line in text.lines() {
        if line.starts_with("## ") {
            if let Some(heading) = current_heading.take() {
                sections.push((heading, current_content.trim().to_string()));
                current_content.clear();
            }
            current_heading = Some(line.trim_start_matches("## ").trim().to_string());
        } else if current_heading.is_some() {
            current_content.push_str(line);
            current_content.push('\n');
        }
    }
    if let Some(heading) = current_heading {
        sections.push((heading, current_content.trim().to_string()));
    }

    format_generate_result(title, &sections, text.len())
}

/// Scan an Obsidian vault and return a summary.
#[instrument(skip(input), fields(vault = %input.vault_path))]
pub async fn handle_obsidian_sync(input: ObsidianSyncInput) -> String {
    if let Err(msg) = validate_vault_path(&input.vault_path) {
        return msg;
    }

    // Discover .md files in the vault
    let vault = Path::new(&input.vault_path);
    let mut notes: Vec<(String, Option<String>, usize)> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(vault) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("md") {
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                // Try to extract title from first H1
                let title = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|content| {
                        content
                            .lines()
                            .find(|l| l.starts_with("# "))
                            .map(|l| l.trim_start_matches("# ").trim().to_string())
                    });

                notes.push((filename, title, 0));
            }
        }
    }

    let count = notes.len();
    format_obsidian_sync_result(count, &notes, &input.vault_path)
}

/// Audit tags for convention violations.
#[instrument(skip(input), fields(tag_count = input.tags.len()))]
pub async fn handle_tag_audit(input: TagAuditInput) -> String {
    let results: Vec<TagAuditEntry> = input
        .tags
        .iter()
        .map(|tag| {
            let mut issues = Vec::new();

            // Check for uppercase characters
            if tag.chars().any(|c| c.is_uppercase()) {
                issues.push("Contains uppercase characters".to_string());
            }

            // Check for invalid separators (should use :: not /)
            if tag.contains('/') {
                issues.push("Uses '/' separator instead of '::'".to_string());
            }

            let suggestion = if !issues.is_empty() {
                Some(tag.to_lowercase().replace('/', "::"))
            } else {
                None
            };

            (tag.clone(), issues, suggestion, Vec::new())
        })
        .collect();

    format_tag_audit_result(&results)
}

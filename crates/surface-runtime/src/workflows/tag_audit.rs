use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde::Serialize;
use taxonomy::{normalize_tag, suggest_tag, validate_tag};

use crate::error::SurfaceError;

#[derive(Debug, Clone, Serialize)]
pub struct TagAuditEntry {
    pub tag: String,
    pub valid: bool,
    pub normalized: String,
    pub suggestion: Option<String>,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TagAuditSummary {
    pub source_file: PathBuf,
    pub applied_fixes: bool,
    pub entries: Vec<TagAuditEntry>,
}

pub struct TagAuditService;

impl Default for TagAuditService {
    fn default() -> Self {
        Self::new()
    }
}

impl TagAuditService {
    pub fn new() -> Self {
        Self
    }

    pub fn audit_file(
        &self,
        file: &Path,
        apply_fixes: bool,
    ) -> Result<TagAuditSummary, SurfaceError> {
        if !file.exists() {
            return Err(SurfaceError::PathNotFound(file.to_path_buf()));
        }
        let content = std::fs::read_to_string(file)?;
        let original_tags: Vec<String> = content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToString::to_string)
            .collect();

        let entries: Vec<TagAuditEntry> = original_tags
            .iter()
            .map(|tag| {
                let normalized = normalize_tag(tag);
                let suggestion = suggest_tag(tag, 3).into_iter().next();
                let validation = validate_tag(tag);
                let mut issues = validation;
                if normalized != *tag {
                    issues.push(format!("normalized form would be `{normalized}`"));
                }
                TagAuditEntry {
                    tag: tag.clone(),
                    valid: issues.is_empty(),
                    normalized,
                    suggestion,
                    issues,
                }
            })
            .collect();

        if apply_fixes {
            let normalized_tags: BTreeSet<String> = entries
                .iter()
                .map(|entry| entry.normalized.clone())
                .collect();
            let rewritten = normalized_tags.into_iter().collect::<Vec<_>>().join("\n");
            std::fs::write(file, format!("{rewritten}\n"))?;
        }

        Ok(TagAuditSummary {
            source_file: file.to_path_buf(),
            applied_fixes: apply_fixes,
            entries,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_audit_service_default_does_not_panic() {
        let _service: TagAuditService = Default::default();
    }
}

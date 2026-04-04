use std::collections::HashMap;
use std::path::Path;

use card::registry::CardRegistry;
use chrono::Utc;
use sha2::{Digest, Sha256};
use tracing::warn;

use crate::error::CardloopError;
use crate::models::{IssueKind, ItemStatus, LoopKind, Tier, WorkItem};
use crate::scanners::Scanner;

/// Scans for cards whose source Obsidian notes have changed since last sync.
///
/// Compares current note file content hashes against stored `content_hash` in
/// the card registry. Emits `StaleContent` work items when hashes differ.
pub struct StaleScanner<'a> {
    registry: &'a CardRegistry,
    vault_root: &'a Path,
}

impl<'a> StaleScanner<'a> {
    pub fn new(registry: &'a CardRegistry, vault_root: &'a Path) -> Self {
        Self {
            registry,
            vault_root,
        }
    }

    fn item_id(source_path: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"stale:");
        hasher.update(source_path.as_bytes());
        let hash = hasher.finalize();
        hash.iter().take(8).map(|b| format!("{b:02x}")).collect()
    }

    /// Compute a content hash for a file's contents.
    /// Uses the same algorithm as `card::slug::compute_content_hash`.
    fn hash_file_content(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.trim().as_bytes());
        let hash = hasher.finalize();
        hash.iter().take(6).map(|b| format!("{b:02x}")).collect()
    }
}

impl Scanner for StaleScanner<'_> {
    fn scan(&self, scan_number: u32) -> Result<Vec<WorkItem>, CardloopError> {
        // Load all notes from registry
        let notes = self.registry.list_notes()?;

        if notes.is_empty() {
            return Ok(Vec::new());
        }

        // Group notes by source_path, keeping their stored content_hash
        let mut note_hashes: HashMap<String, (String, Option<String>)> = HashMap::new();
        for note in &notes {
            if let Some(ref stored_hash) = note.content_hash {
                note_hashes.insert(
                    note.source_path.clone(),
                    (note.note_id.clone(), Some(stored_hash.clone())),
                );
            }
        }

        let mut items = Vec::new();
        let now = Utc::now();

        for (source_path, (note_id, stored_hash)) in &note_hashes {
            let stored_hash = match stored_hash {
                Some(h) => h,
                None => continue, // No stored hash to compare
            };

            // Resolve full path
            let full_path = self.vault_root.join(source_path);

            // Check if source file still exists
            if !full_path.exists() {
                // Source deleted -- emit as orphan
                let id = Self::item_id(source_path);
                items.push(WorkItem {
                    id,
                    loop_kind: LoopKind::Audit,
                    issue_kind: IssueKind::StaleContent,
                    tier: Tier::QuickFix,
                    status: ItemStatus::Open,
                    slug: None,
                    source_path: source_path.clone(),
                    summary: format!("Source note deleted: {source_path}"),
                    detail: Some(format!(
                        "note_id={note_id}, stored_hash={stored_hash}, file missing"
                    )),
                    first_seen: now,
                    resolved_at: None,
                    attestation: None,
                    scan_number,
                    cluster_id: None,
                    confidence: Some(0.95),
                });
                continue;
            }

            // Read current content and compute hash
            let content = match std::fs::read_to_string(&full_path) {
                Ok(c) => c,
                Err(e) => {
                    warn!(path = %full_path.display(), error = %e, "Failed to read note file");
                    continue;
                }
            };

            let current_hash = Self::hash_file_content(&content);

            if current_hash != *stored_hash {
                let id = Self::item_id(source_path);
                items.push(WorkItem {
                    id,
                    loop_kind: LoopKind::Audit,
                    issue_kind: IssueKind::StaleContent,
                    tier: Tier::QuickFix,
                    status: ItemStatus::Open,
                    slug: None,
                    source_path: source_path.clone(),
                    summary: format!("Source note modified since last sync: {source_path}"),
                    detail: Some(format!(
                        "note_id={note_id}, stored_hash={stored_hash}, current_hash={current_hash}"
                    )),
                    first_seen: now,
                    resolved_at: None,
                    attestation: None,
                    scan_number,
                    cluster_id: None,
                    confidence: Some(0.9),
                });
            }
        }

        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn item_id_is_deterministic() {
        let id1 = StaleScanner::item_id("notes/rust.md");
        let id2 = StaleScanner::item_id("notes/rust.md");
        assert_eq!(id1, id2);
    }

    #[test]
    fn item_id_differs_by_path() {
        let id1 = StaleScanner::item_id("notes/rust.md");
        let id2 = StaleScanner::item_id("notes/python.md");
        assert_ne!(id1, id2);
    }

    #[test]
    fn hash_changes_on_content_change() {
        let h1 = StaleScanner::hash_file_content("Hello world");
        let h2 = StaleScanner::hash_file_content("Hello world updated");
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_ignores_whitespace_trim() {
        let h1 = StaleScanner::hash_file_content("Hello world");
        let h2 = StaleScanner::hash_file_content("  Hello world  ");
        assert_eq!(h1, h2);
    }

    #[test]
    fn scan_empty_registry() {
        let dir = TempDir::new().unwrap();
        let registry = CardRegistry::open(":memory:").unwrap();
        let scanner = StaleScanner::new(&registry, dir.path());
        let items = scanner.scan(1).unwrap();
        assert!(items.is_empty());
    }
}

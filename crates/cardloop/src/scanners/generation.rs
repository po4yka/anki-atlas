use std::collections::HashSet;
use std::path::Path;

use chrono::Utc;

use card::registry::CardRegistry;

use crate::error::CardloopError;
use crate::models::{IssueKind, ItemStatus, LoopKind, Tier, WorkItem};
use crate::scanners::Scanner;

/// Default glob patterns for discovering Obsidian notes.
const NOTE_PATTERNS: &[&str] = &["*.md"];
/// Directories to ignore when discovering notes.
const IGNORE_DIRS: &[&str] = &[".obsidian", ".trash", "templates", "node_modules"];

/// Scans Obsidian vault notes and cross-references with `CardRegistry`
/// to find uncovered topics (notes with no cards).
pub struct GenerationScanner<'a> {
    registry: &'a CardRegistry,
    vault_root: &'a Path,
}

impl<'a> GenerationScanner<'a> {
    pub fn new(registry: &'a CardRegistry, vault_root: &'a Path) -> Self {
        Self {
            registry,
            vault_root,
        }
    }

    /// Generate a deterministic ID for a note source path.
    fn item_id(source_path: &str, discriminator: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"gen:");
        hasher.update(source_path.as_bytes());
        hasher.update(b":");
        hasher.update(discriminator.as_bytes());
        let hash = hasher.finalize();
        hash.iter().take(8).map(|b| format!("{b:02x}")).collect()
    }
}

impl Scanner for GenerationScanner<'_> {
    fn scan(&self, scan_number: u32) -> Result<Vec<WorkItem>, CardloopError> {
        let note_paths = obsidian::discover_notes(self.vault_root, NOTE_PATTERNS, IGNORE_DIRS)
            .map_err(|e| CardloopError::Validation(e.to_string()))?;

        // Collect all source_paths that already have cards in the registry.
        let all_cards = self.registry.find_cards(None, None, None)?;
        let covered_sources: HashSet<String> =
            all_cards.iter().map(|c| c.source_path.clone()).collect();

        // Canonicalize vault root for consistent path stripping
        // (discover_notes returns canonicalized paths).
        let canonical_root = self
            .vault_root
            .canonicalize()
            .unwrap_or_else(|_| self.vault_root.to_path_buf());

        let now = Utc::now();
        let mut items = Vec::new();

        for note_path in &note_paths {
            let relative = note_path.strip_prefix(&canonical_root).unwrap_or(note_path);
            let source_str = relative.to_string_lossy().to_string();

            if covered_sources.contains(&source_str) {
                continue;
            }

            // Parse the note to get a title for the summary.
            let title = obsidian::parse_note(note_path, Some(&canonical_root))
                .ok()
                .and_then(|n| n.title)
                .unwrap_or_else(|| source_str.clone());

            let id = GenerationScanner::item_id(&source_str, "uncovered");
            items.push(WorkItem {
                id,
                loop_kind: LoopKind::Generation,
                issue_kind: IssueKind::UncoveredTopic {
                    topic: title.clone(),
                },
                tier: Tier::Rework,
                status: ItemStatus::Open,
                slug: None,
                source_path: source_str,
                summary: format!("Uncovered note: {title}"),
                detail: None,
                first_seen: now,
                resolved_at: None,
                attestation: None,
                scan_number,
            });
        }

        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use card::registry::CardRegistry;
    use std::fs;

    fn test_registry() -> CardRegistry {
        CardRegistry::open(":memory:").unwrap()
    }

    #[test]
    fn detects_uncovered_notes() {
        let dir = tempfile::tempdir().unwrap();
        let vault = dir.path();

        // Create two markdown notes
        fs::write(
            vault.join("note1.md"),
            "# Ownership\nContent about ownership",
        )
        .unwrap();
        fs::write(
            vault.join("note2.md"),
            "# Borrowing\nContent about borrowing",
        )
        .unwrap();

        let registry = test_registry();
        let scanner = GenerationScanner::new(&registry, vault);
        let items = scanner.scan(1).unwrap();

        assert_eq!(items.len(), 2);
        assert!(items.iter().all(|i| i.loop_kind == LoopKind::Generation));
        assert!(
            items
                .iter()
                .all(|i| matches!(i.issue_kind, IssueKind::UncoveredTopic { .. }))
        );
    }

    #[test]
    fn skips_covered_notes() {
        let dir = tempfile::tempdir().unwrap();
        let vault = dir.path();

        fs::write(vault.join("note1.md"), "# Ownership\nContent").unwrap();
        fs::write(vault.join("note2.md"), "# Borrowing\nContent").unwrap();

        let registry = test_registry();

        // Add a card for note1
        let now = Utc::now();
        registry
            .add_card(&card::CardEntry {
                slug: "ownership-what".into(),
                note_id: "n1".into(),
                source_path: "note1.md".into(),
                front: "What is ownership?".into(),
                back: "A set of rules for memory management.".into(),
                content_hash: "abc".into(),
                metadata_hash: "def".into(),
                language: "en".into(),
                tags: vec![],
                anki_note_id: None,
                created_at: Some(now),
                updated_at: Some(now),
                synced_at: None,
            })
            .unwrap();

        let scanner = GenerationScanner::new(&registry, vault);
        let items = scanner.scan(1).unwrap();

        // Only note2 should be flagged
        assert_eq!(items.len(), 1);
        assert!(items[0].summary.contains("Borrowing"));
    }

    #[test]
    fn empty_vault_produces_no_items() {
        let dir = tempfile::tempdir().unwrap();
        let registry = test_registry();
        let scanner = GenerationScanner::new(&registry, dir.path());
        let items = scanner.scan(1).unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn deterministic_ids() {
        let id1 = GenerationScanner::item_id("notes/rust.md", "uncovered");
        let id2 = GenerationScanner::item_id("notes/rust.md", "uncovered");
        assert_eq!(id1, id2);

        let id3 = GenerationScanner::item_id("notes/go.md", "uncovered");
        assert_ne!(id1, id3);
    }
}

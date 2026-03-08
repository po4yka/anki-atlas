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
    pub broken_links: Vec<(String, String)>,
}

/// Analyze vault structure: wikilinks, orphans, broken links.
#[allow(dead_code)]
pub struct VaultAnalyzer {
    vault_path: PathBuf,
}

impl VaultAnalyzer {
    pub fn new(_vault_path: &Path) -> Self {
        todo!()
    }

    /// Get wikilink targets from a specific note.
    pub fn get_wikilinks(&mut self, _path: &Path) -> Vec<String> {
        todo!()
    }

    /// Find notes with no incoming or outgoing links.
    pub fn find_orphaned(&mut self) -> Vec<PathBuf> {
        todo!()
    }

    /// Compute comprehensive vault statistics.
    pub fn analyze(&mut self) -> VaultStats {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Helper: create a vault with notes. Each entry is (relative_path, content).
    fn create_vault(entries: &[(&str, &str)]) -> TempDir {
        let dir = TempDir::new().unwrap();
        for (rel_path, content) in entries {
            let full = dir.path().join(rel_path);
            if let Some(parent) = full.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&full, content).unwrap();
        }
        dir
    }

    // ── get_wikilinks ──────────────────────────────────────────────

    #[test]
    fn get_wikilinks_basic() {
        let vault = create_vault(&[("note.md", "See [[target]] for info.")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let links = analyzer.get_wikilinks(&vault.path().join("note.md"));
        assert_eq!(links, vec!["target"]);
    }

    #[test]
    fn get_wikilinks_with_alias() {
        let vault = create_vault(&[("note.md", "Read [[target|display text]] here.")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let links = analyzer.get_wikilinks(&vault.path().join("note.md"));
        assert_eq!(links, vec!["target"]);
    }

    #[test]
    fn get_wikilinks_multiple() {
        let vault = create_vault(&[(
            "note.md",
            "Links: [[alpha]], [[beta|b]], and [[gamma]].",
        )]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let links = analyzer.get_wikilinks(&vault.path().join("note.md"));
        assert_eq!(links, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn get_wikilinks_none() {
        let vault = create_vault(&[("note.md", "No links here.")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let links = analyzer.get_wikilinks(&vault.path().join("note.md"));
        assert!(links.is_empty());
    }

    #[test]
    fn get_wikilinks_unknown_note_returns_empty() {
        let vault = create_vault(&[("note.md", "Some content.")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let links = analyzer.get_wikilinks(&vault.path().join("nonexistent.md"));
        assert!(links.is_empty());
    }

    #[test]
    fn get_wikilinks_strips_whitespace_in_target() {
        let vault = create_vault(&[("note.md", "See [[ spaced ]] here.")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let links = analyzer.get_wikilinks(&vault.path().join("note.md"));
        assert_eq!(links, vec!["spaced"]);
    }

    // ── find_orphaned ──────────────────────────────────────────────

    #[test]
    fn find_orphaned_no_links_at_all() {
        let vault = create_vault(&[
            ("a.md", "Standalone note A."),
            ("b.md", "Standalone note B."),
        ]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let orphaned = analyzer.find_orphaned();
        // Both are orphaned (no outgoing, no incoming)
        let mut stems: Vec<_> = orphaned
            .iter()
            .map(|p| p.file_stem().unwrap().to_str().unwrap().to_string())
            .collect();
        stems.sort();
        assert_eq!(stems, vec!["a", "b"]);
    }

    #[test]
    fn find_orphaned_linked_notes_not_orphaned() {
        let vault = create_vault(&[
            ("a.md", "See [[b]]."),
            ("b.md", "See [[a]]."),
        ]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let orphaned = analyzer.find_orphaned();
        assert!(orphaned.is_empty());
    }

    #[test]
    fn find_orphaned_mixed() {
        let vault = create_vault(&[
            ("a.md", "See [[b]]."),
            ("b.md", "Linked by a."),
            ("c.md", "Completely alone."),
        ]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let orphaned = analyzer.find_orphaned();
        let stems: Vec<_> = orphaned
            .iter()
            .map(|p| p.file_stem().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(stems, vec!["c"]);
    }

    #[test]
    fn find_orphaned_incoming_only_not_orphaned() {
        // b has no outgoing links but is linked by a -> not orphaned
        let vault = create_vault(&[
            ("a.md", "See [[b]]."),
            ("b.md", "I have no links."),
        ]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let orphaned = analyzer.find_orphaned();
        // a has outgoing (not orphaned), b has incoming (not orphaned)
        assert!(orphaned.is_empty());
    }

    // ── analyze ────────────────────────────────────────────────────

    #[test]
    fn analyze_total_notes() {
        let vault = create_vault(&[
            ("a.md", "Note A."),
            ("b.md", "Note B."),
            ("sub/c.md", "Note C."),
        ]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let stats = analyzer.analyze();
        assert_eq!(stats.total_notes, 3);
    }

    #[test]
    fn analyze_total_dirs() {
        let vault = create_vault(&[
            ("a.md", "Note."),
            ("sub/b.md", "Note."),
            ("sub/deep/c.md", "Note."),
        ]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let stats = analyzer.analyze();
        // Three distinct parent dirs: root, sub, sub/deep
        assert_eq!(stats.total_dirs, 3);
    }

    #[test]
    fn analyze_notes_with_frontmatter() {
        let vault = create_vault(&[
            ("a.md", "---\ntitle: A\n---\nBody."),
            ("b.md", "No frontmatter."),
            ("c.md", "---\ntags: [x]\n---\nBody."),
        ]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let stats = analyzer.analyze();
        assert_eq!(stats.notes_with_frontmatter, 2);
    }

    #[test]
    fn analyze_wikilinks_count() {
        let vault = create_vault(&[
            ("a.md", "See [[b]] and [[c]]."),
            ("b.md", "See [[a]]."),
            ("c.md", "No links."),
        ]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let stats = analyzer.analyze();
        assert_eq!(stats.wikilinks_count, 3);
    }

    #[test]
    fn analyze_broken_links() {
        let vault = create_vault(&[
            ("a.md", "See [[b]] and [[missing]]."),
            ("b.md", "Content."),
        ]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let stats = analyzer.analyze();
        assert_eq!(stats.broken_links, vec![("a".to_string(), "missing".to_string())]);
    }

    #[test]
    fn analyze_orphaned_notes() {
        let vault = create_vault(&[
            ("a.md", "See [[b]]."),
            ("b.md", "Content."),
            ("c.md", "Alone."),
        ]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let stats = analyzer.analyze();
        assert_eq!(stats.orphaned_notes, vec!["c"]);
    }

    #[test]
    fn analyze_empty_vault() {
        let vault = TempDir::new().unwrap();
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let stats = analyzer.analyze();
        assert_eq!(stats.total_notes, 0);
        assert_eq!(stats.total_dirs, 0);
        assert_eq!(stats.wikilinks_count, 0);
        assert!(stats.orphaned_notes.is_empty());
        assert!(stats.broken_links.is_empty());
    }

    #[test]
    fn analyze_lazy_scan_idempotent() {
        let vault = create_vault(&[("a.md", "See [[b]].")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        // Call multiple methods - scan should happen once
        let _ = analyzer.get_wikilinks(&vault.path().join("a.md"));
        let stats = analyzer.analyze();
        assert_eq!(stats.total_notes, 1);
    }

    // ── Send + Sync compile-time assertion ─────────────────────────

    #[test]
    fn vault_stats_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VaultStats>();
    }

    #[test]
    fn vault_analyzer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VaultAnalyzer>();
    }
}

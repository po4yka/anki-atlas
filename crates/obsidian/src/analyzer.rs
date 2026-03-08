use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::parser::{discover_notes, parse_note, DEFAULT_IGNORE_DIRS};

/// Regex matching wikilinks: `[[target]]` or `[[target|alias]]`.
static WIKILINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]").unwrap());

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

/// Cached scan data for a vault.
struct ScanData {
    /// Map from note stem -> list of wikilink targets.
    links: HashMap<String, Vec<String>>,
    /// Map from note stem -> full path.
    paths: HashMap<String, PathBuf>,
    /// Set of note stems that have frontmatter.
    has_frontmatter: HashSet<String>,
    /// Set of distinct parent directories.
    dirs: HashSet<PathBuf>,
}

/// Analyze vault structure: wikilinks, orphans, broken links.
pub struct VaultAnalyzer {
    vault_path: PathBuf,
    scan: Option<ScanData>,
}

impl VaultAnalyzer {
    pub fn new(vault_path: &Path) -> Self {
        Self {
            vault_path: vault_path.to_path_buf(),
            scan: None,
        }
    }

    /// Lazy scan: discover and parse all notes on first access.
    fn ensure_scanned(&mut self) {
        if self.scan.is_some() {
            return;
        }

        let mut links: HashMap<String, Vec<String>> = HashMap::new();
        let mut paths: HashMap<String, PathBuf> = HashMap::new();
        let mut has_frontmatter: HashSet<String> = HashSet::new();
        let mut dirs: HashSet<PathBuf> = HashSet::new();

        let note_paths = discover_notes(&self.vault_path, &["*.md"], DEFAULT_IGNORE_DIRS)
            .unwrap_or_default();

        for note_path in &note_paths {
            let stem = note_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();

            paths.insert(stem.clone(), note_path.clone());

            if let Some(parent) = note_path.parent() {
                dirs.insert(parent.to_path_buf());
            }

            if let Ok(parsed) = parse_note(note_path, Some(&self.vault_path)) {
                if !parsed.frontmatter.is_empty() {
                    has_frontmatter.insert(stem.clone());
                }
                let wikilinks = extract_wikilinks(&parsed.content);
                links.insert(stem, wikilinks);
            }
        }

        self.scan = Some(ScanData {
            links,
            paths,
            has_frontmatter,
            dirs,
        });
    }

    /// Get wikilink targets from a specific note.
    pub fn get_wikilinks(&mut self, path: &Path) -> Vec<String> {
        self.ensure_scanned();
        let scan = self.scan.as_ref().unwrap();

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");

        scan.links.get(stem).cloned().unwrap_or_default()
    }

    /// Find notes with no incoming or outgoing links.
    pub fn find_orphaned(&mut self) -> Vec<PathBuf> {
        self.ensure_scanned();
        let scan = self.scan.as_ref().unwrap();

        // Build set of notes that have incoming links
        let mut has_incoming: HashSet<&str> = HashSet::new();
        for targets in scan.links.values() {
            for target in targets {
                has_incoming.insert(target.as_str());
            }
        }

        let mut orphaned: Vec<PathBuf> = scan
            .paths
            .iter()
            .filter(|(stem, _)| {
                let outgoing = scan.links.get(stem.as_str()).is_none_or(Vec::is_empty);
                let incoming = has_incoming.contains(stem.as_str());
                outgoing && !incoming
            })
            .map(|(_, path)| path.clone())
            .collect();

        orphaned.sort();
        orphaned
    }

    /// Compute comprehensive vault statistics.
    pub fn analyze(&mut self) -> VaultStats {
        self.ensure_scanned();
        let scan = self.scan.as_ref().unwrap();

        let total_notes = scan.paths.len();
        let total_dirs = scan.dirs.len();
        let notes_with_frontmatter = scan.has_frontmatter.len();

        let mut wikilinks_count = 0;
        let mut broken_links = Vec::new();

        for (stem, targets) in &scan.links {
            wikilinks_count += targets.len();
            for target in targets {
                if !scan.paths.contains_key(target) {
                    broken_links.push((stem.clone(), target.clone()));
                }
            }
        }

        let orphaned = self.find_orphaned();
        let orphaned_notes: Vec<String> = orphaned
            .iter()
            .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(String::from))
            .collect();

        VaultStats {
            total_notes,
            total_dirs,
            notes_with_frontmatter,
            wikilinks_count,
            orphaned_notes,
            broken_links,
        }
    }
}

/// Extract wikilink targets from content, trimming whitespace.
fn extract_wikilinks(content: &str) -> Vec<String> {
    WIKILINK_RE
        .captures_iter(content)
        .map(|c| c[1].trim().to_string())
        .collect()
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

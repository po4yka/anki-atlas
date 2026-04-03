use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::ObsidianError;
use crate::parser::{DEFAULT_IGNORE_DIRS, discover_notes, parse_note};

/// A broken wikilink: the source note path and the unresolved target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrokenLink {
    pub source: String,
    pub target: String,
}

/// Regex matching wikilinks: `[[target]]` or `[[target|alias]]`.
static WIKILINK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]").unwrap());

/// Statistics about an Obsidian vault.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VaultStats {
    pub total_notes: usize,
    pub total_dirs: usize,
    pub notes_with_frontmatter: usize,
    pub wikilinks_count: usize,
    pub orphaned_notes: Vec<String>,
    pub broken_links: Vec<BrokenLink>,
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

impl ScanData {
    /// Notes with no outgoing AND no incoming links, sorted by path.
    fn orphaned_paths(&self) -> Vec<PathBuf> {
        let has_incoming: HashSet<&str> =
            self.links.values().flatten().map(String::as_str).collect();

        let mut orphaned: Vec<PathBuf> = self
            .paths
            .iter()
            .filter(|(stem, _)| {
                let outgoing = self.links.get(stem.as_str()).is_none_or(Vec::is_empty);
                let incoming = has_incoming.contains(stem.as_str());
                outgoing && !incoming
            })
            .map(|(_, path)| path.clone())
            .collect();

        orphaned.sort();
        orphaned
    }
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
    fn ensure_scanned(&mut self) -> Result<(), ObsidianError> {
        if self.scan.is_some() {
            return Ok(());
        }

        let mut links: HashMap<String, Vec<String>> = HashMap::new();
        let mut paths: HashMap<String, PathBuf> = HashMap::new();
        let mut has_frontmatter: HashSet<String> = HashSet::new();
        let mut dirs: HashSet<PathBuf> = HashSet::new();

        let note_paths = discover_notes(&self.vault_path, &["*.md"], DEFAULT_IGNORE_DIRS)?;

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

            let parsed = parse_note(note_path, Some(&self.vault_path))?;
            if !parsed.frontmatter.is_empty() {
                has_frontmatter.insert(stem.clone());
            }
            let wikilinks = extract_wikilinks(&parsed.content);
            links.insert(stem, wikilinks);
        }

        self.scan = Some(ScanData {
            links,
            paths,
            has_frontmatter,
            dirs,
        });

        Ok(())
    }

    /// Ensure vault is scanned and return a reference to cached data.
    fn scan_data(&mut self) -> Result<&ScanData, ObsidianError> {
        self.ensure_scanned()?;
        Ok(self.scan.as_ref().unwrap())
    }

    /// Get wikilink targets from a specific note.
    pub fn get_wikilinks(&mut self, path: &Path) -> Result<Vec<String>, ObsidianError> {
        let scan = self.scan_data()?;
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");

        Ok(scan.links.get(stem).cloned().unwrap_or_default())
    }

    /// Find notes with no incoming or outgoing links.
    pub fn find_orphaned(&mut self) -> Result<Vec<PathBuf>, ObsidianError> {
        Ok(self.scan_data()?.orphaned_paths())
    }

    /// Compute comprehensive vault statistics.
    pub fn analyze(&mut self) -> Result<VaultStats, ObsidianError> {
        let scan = self.scan_data()?;

        let wikilinks_count: usize = scan.links.values().map(Vec::len).sum();

        let broken_links: Vec<BrokenLink> = scan
            .links
            .iter()
            .flat_map(|(stem, targets)| {
                targets
                    .iter()
                    .filter(|t| !scan.paths.contains_key(t.as_str()))
                    .map(move |t| BrokenLink {
                        source: stem.clone(),
                        target: t.clone(),
                    })
            })
            .collect();

        let orphaned_notes: Vec<String> = scan
            .orphaned_paths()
            .iter()
            .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(String::from))
            .collect();

        Ok(VaultStats {
            total_notes: scan.paths.len(),
            total_dirs: scan.dirs.len(),
            notes_with_frontmatter: scan.has_frontmatter.len(),
            wikilinks_count,
            orphaned_notes,
            broken_links,
        })
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
    use crate::ObsidianError;
    use std::fs;
    use std::path::Path;
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
        let links = analyzer
            .get_wikilinks(&vault.path().join("note.md"))
            .unwrap();
        assert_eq!(links, vec!["target"]);
    }

    #[test]
    fn get_wikilinks_with_alias() {
        let vault = create_vault(&[("note.md", "Read [[target|display text]] here.")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let links = analyzer
            .get_wikilinks(&vault.path().join("note.md"))
            .unwrap();
        assert_eq!(links, vec!["target"]);
    }

    #[test]
    fn get_wikilinks_multiple() {
        let vault = create_vault(&[("note.md", "Links: [[alpha]], [[beta|b]], and [[gamma]].")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let links = analyzer
            .get_wikilinks(&vault.path().join("note.md"))
            .unwrap();
        assert_eq!(links, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn get_wikilinks_none() {
        let vault = create_vault(&[("note.md", "No links here.")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let links = analyzer
            .get_wikilinks(&vault.path().join("note.md"))
            .unwrap();
        assert!(links.is_empty());
    }

    #[test]
    fn get_wikilinks_unknown_note_returns_empty() {
        let vault = create_vault(&[("note.md", "Some content.")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let links = analyzer
            .get_wikilinks(&vault.path().join("nonexistent.md"))
            .unwrap();
        assert!(links.is_empty());
    }

    #[test]
    fn get_wikilinks_strips_whitespace_in_target() {
        let vault = create_vault(&[("note.md", "See [[ spaced ]] here.")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let links = analyzer
            .get_wikilinks(&vault.path().join("note.md"))
            .unwrap();
        assert_eq!(links, vec!["spaced"]);
    }

    #[test]
    fn get_wikilinks_missing_vault_returns_error() {
        let mut analyzer = VaultAnalyzer::new(Path::new("/nonexistent/vault"));
        let error = analyzer
            .get_wikilinks(Path::new("/nonexistent/vault/note.md"))
            .unwrap_err();
        assert!(matches!(error, ObsidianError::NotFound(_)));
    }

    // ── find_orphaned ──────────────────────────────────────────────

    #[test]
    fn find_orphaned_no_links_at_all() {
        let vault = create_vault(&[
            ("a.md", "Standalone note A."),
            ("b.md", "Standalone note B."),
        ]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let orphaned = analyzer.find_orphaned().unwrap();
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
        let vault = create_vault(&[("a.md", "See [[b]]."), ("b.md", "See [[a]].")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let orphaned = analyzer.find_orphaned().unwrap();
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
        let orphaned = analyzer.find_orphaned().unwrap();
        let stems: Vec<_> = orphaned
            .iter()
            .map(|p| p.file_stem().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(stems, vec!["c"]);
    }

    #[test]
    fn find_orphaned_incoming_only_not_orphaned() {
        // b has no outgoing links but is linked by a -> not orphaned
        let vault = create_vault(&[("a.md", "See [[b]]."), ("b.md", "I have no links.")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let orphaned = analyzer.find_orphaned().unwrap();
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
        let stats = analyzer.analyze().unwrap();
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
        let stats = analyzer.analyze().unwrap();
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
        let stats = analyzer.analyze().unwrap();
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
        let stats = analyzer.analyze().unwrap();
        assert_eq!(stats.wikilinks_count, 3);
    }

    #[test]
    fn analyze_broken_links() {
        let vault = create_vault(&[("a.md", "See [[b]] and [[missing]]."), ("b.md", "Content.")]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let stats = analyzer.analyze().unwrap();
        assert_eq!(
            stats.broken_links,
            vec![BrokenLink {
                source: "a".to_string(),
                target: "missing".to_string(),
            }]
        );
    }

    #[test]
    fn analyze_orphaned_notes() {
        let vault = create_vault(&[
            ("a.md", "See [[b]]."),
            ("b.md", "Content."),
            ("c.md", "Alone."),
        ]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let stats = analyzer.analyze().unwrap();
        assert_eq!(stats.orphaned_notes, vec!["c"]);
    }

    #[test]
    fn analyze_empty_vault() {
        let vault = TempDir::new().unwrap();
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let stats = analyzer.analyze().unwrap();
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
        let _ = analyzer.get_wikilinks(&vault.path().join("a.md")).unwrap();
        let stats = analyzer.analyze().unwrap();
        assert_eq!(stats.total_notes, 1);
    }

    #[test]
    fn analyze_invalid_note_returns_error() {
        let vault = create_vault(&[
            ("good.md", "# Good\nBody\n"),
            ("broken.md", "---\n: invalid yaml\n---\nBody\n"),
        ]);
        let mut analyzer = VaultAnalyzer::new(vault.path());
        let error = analyzer.analyze().unwrap_err();
        assert!(matches!(error, ObsidianError::Yaml(_)));
    }

    // ── Send + Sync compile-time assertion ─────────────────────────

    common::assert_send_sync!(VaultStats, VaultAnalyzer);
}

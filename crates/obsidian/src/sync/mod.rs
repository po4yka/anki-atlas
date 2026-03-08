use std::path::{Path, PathBuf};

use crate::parser::{DEFAULT_IGNORE_DIRS, ParsedNote, discover_notes, parse_note};

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
    pub fn merge(self, other: SyncResult) -> SyncResult {
        SyncResult {
            generated: self.generated + other.generated,
            updated: self.updated + other.updated,
            skipped: self.skipped + other.skipped,
            failed: self.failed + other.failed,
            errors: self.errors.into_iter().chain(other.errors).collect(),
        }
    }
}

/// Placeholder for generated card reference.
#[derive(Debug, Clone)]
pub struct GeneratedCardRef {
    pub slug: String,
    pub apf_html: String,
}

/// Trait for card generation from a parsed note (injected dependency).
pub trait CardGenerator: Send + Sync {
    fn generate(&self, note: &ParsedNote) -> Vec<GeneratedCardRef>;
}

/// Orchestrates: discover notes -> generate cards -> validate -> aggregate.
pub struct ObsidianSyncWorkflow<G: CardGenerator> {
    generator: G,
    on_progress: Option<ProgressCallback>,
}

impl<G: CardGenerator> ObsidianSyncWorkflow<G> {
    pub fn new(generator: G, on_progress: Option<ProgressCallback>) -> Self {
        Self {
            generator,
            on_progress,
        }
    }

    /// Discover and parse all notes in vault.
    pub fn scan_vault(&self, vault_path: &Path, source_dirs: Option<&[&str]>) -> Vec<ParsedNote> {
        let dirs_to_scan: Vec<PathBuf> = match source_dirs {
            Some(dirs) => dirs
                .iter()
                .map(|d| vault_path.join(d))
                .filter(|p| p.exists())
                .collect(),
            None => vec![vault_path.to_path_buf()],
        };

        dirs_to_scan
            .iter()
            .flat_map(|dir| discover_notes(dir, &["*.md"], DEFAULT_IGNORE_DIRS).unwrap_or_default())
            .filter_map(|path| parse_note(&path, Some(vault_path)).ok())
            .collect()
    }

    /// Full pipeline: scan -> process all notes -> aggregate results.
    pub fn run(&self, vault_path: &Path, source_dirs: Option<&[&str]>) -> SyncResult {
        let notes = self.scan_vault(vault_path, source_dirs);
        let total = notes.len();
        let mut result = SyncResult::default();

        for (i, note) in notes.iter().enumerate() {
            if let Some(cb) = &self.on_progress {
                cb("generating", i + 1, total);
            }

            let cards = self.generator.generate(note);
            if cards.is_empty() {
                result.failed += 1;
            } else {
                result.generated += cards.len();
            }
        }

        result
    }
}

#[cfg(test)]
mod tests;

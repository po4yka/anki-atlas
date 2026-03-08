#![allow(unused, unreachable_code)]

use std::path::{Path, PathBuf};

use crate::parser::ParsedNote;

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
        todo!()
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
        todo!()
    }

    /// Discover and parse all notes in vault.
    pub fn scan_vault(
        &self,
        vault_path: &Path,
        source_dirs: Option<&[&str]>,
    ) -> Vec<ParsedNote> {
        todo!()
    }

    /// Full pipeline: scan -> process all notes -> aggregate results.
    pub fn run(&self, vault_path: &Path, source_dirs: Option<&[&str]>) -> SyncResult {
        todo!()
    }
}

#[cfg(test)]
mod tests;

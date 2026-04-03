use std::path::{Path, PathBuf};
use std::sync::Arc;

use obsidian::analyzer::VaultAnalyzer;
use obsidian::sync::ObsidianSyncWorkflow;
use serde::Serialize;

use super::preview::PreviewCardGenerator;
use super::progress::{SurfaceOperation, SurfaceProgressSink, emit_progress};
use crate::error::SurfaceError;

#[derive(Debug, Clone, Serialize)]
pub struct ObsidianNotePreview {
    pub path: PathBuf,
    pub title: Option<String>,
    pub sections: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ObsidianScanPreview {
    pub vault_path: PathBuf,
    pub source_dirs: Vec<String>,
    pub note_count: usize,
    pub generated_cards: usize,
    pub orphaned_notes: Vec<String>,
    pub broken_links: Vec<obsidian::BrokenLink>,
    pub notes: Vec<ObsidianNotePreview>,
}

pub struct ObsidianScanService;

impl Default for ObsidianScanService {
    fn default() -> Self {
        Self::new()
    }
}

impl ObsidianScanService {
    pub fn new() -> Self {
        Self
    }

    pub fn scan(
        &self,
        vault: &Path,
        source_dirs: &[String],
        execution_mode: common::ExecutionMode,
    ) -> Result<ObsidianScanPreview, SurfaceError> {
        self.scan_with_progress(vault, source_dirs, execution_mode, None)
    }

    pub fn scan_with_progress(
        &self,
        vault: &Path,
        source_dirs: &[String],
        execution_mode: common::ExecutionMode,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<ObsidianScanPreview, SurfaceError> {
        if execution_mode == common::ExecutionMode::Execute {
            return Err(SurfaceError::Unsupported(
                "obsidian persistence is not implemented; use --dry-run".to_string(),
            ));
        }
        if !vault.exists() {
            return Err(SurfaceError::PathNotFound(vault.to_path_buf()));
        }

        emit_progress(
            progress.as_ref(),
            SurfaceOperation::ObsidianScan,
            "scanning_vault",
            0,
            1,
            format!("scanning vault {}", vault.display()),
        );
        let workflow = ObsidianSyncWorkflow::new(
            PreviewCardGenerator,
            progress.as_ref().map(|sink| {
                let sink = Arc::clone(sink);
                Box::new(move |phase: &str, current: usize, total: usize| {
                    sink(super::progress::SurfaceProgressEvent {
                        operation: SurfaceOperation::ObsidianScan,
                        stage: phase.to_string(),
                        current,
                        total,
                        message: format!("{phase}: {current}/{total}"),
                    });
                }) as obsidian::sync::ProgressCallback
            }),
        );
        let dir_refs: Vec<&str> = source_dirs.iter().map(String::as_str).collect();
        let notes = if dir_refs.is_empty() {
            workflow.scan_vault(vault, None)?
        } else {
            workflow.scan_vault(vault, Some(&dir_refs))?
        };
        let sync = if dir_refs.is_empty() {
            workflow.run(vault, None)?
        } else {
            workflow.run(vault, Some(&dir_refs))?
        };

        emit_progress(
            progress.as_ref(),
            SurfaceOperation::ObsidianScan,
            "analyzing_vault",
            0,
            1,
            "analyzing vault structure",
        );
        let mut analyzer = VaultAnalyzer::new(vault);
        let stats = analyzer.analyze()?;
        let note_previews = notes
            .into_iter()
            .map(|note| ObsidianNotePreview {
                path: note.path,
                title: note.title,
                sections: note.sections.len(),
            })
            .collect();

        let preview = ObsidianScanPreview {
            vault_path: vault.to_path_buf(),
            source_dirs: source_dirs.to_vec(),
            note_count: stats.total_notes,
            generated_cards: sync.generated,
            orphaned_notes: stats.orphaned_notes,
            broken_links: stats.broken_links,
            notes: note_previews,
        };
        emit_progress(
            progress.as_ref(),
            SurfaceOperation::ObsidianScan,
            "completed",
            preview.note_count,
            preview.note_count.max(1),
            format!(
                "obsidian scan completed with {} generated cards",
                preview.generated_cards
            ),
        );
        Ok(preview)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obsidian_scan_service_default_does_not_panic() {
        let _service: ObsidianScanService = Default::default();
    }
}

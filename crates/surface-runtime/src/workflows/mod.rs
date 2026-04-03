mod indexing;
mod obsidian_scan;
mod preview;
mod progress;
mod sync;
mod tag_audit;
mod validation;

pub use indexing::{IndexExecutionSummary, IndexExecutor, IndexingService};
pub use obsidian_scan::{ObsidianNotePreview, ObsidianScanPreview, ObsidianScanService};
pub use preview::{GeneratePreview, GeneratePreviewService};
pub use progress::{SurfaceOperation, SurfaceProgressEvent, SurfaceProgressSink};
pub use sync::{SyncExecutionHandle, SyncExecutionService, SyncExecutionSummary, SyncStatsSummary};
pub use tag_audit::{TagAuditEntry, TagAuditService, TagAuditSummary};
pub use validation::{QualityCheck, ValidationService, ValidationSummary};

#[cfg(test)]
mod tests {
    use super::*;

    common::assert_send_sync!(
        GeneratePreview,
        ValidationSummary,
        ObsidianNotePreview,
        ObsidianScanPreview,
        TagAuditEntry,
        TagAuditSummary,
        SyncExecutionSummary,
        IndexExecutionSummary,
        SyncStatsSummary,
        GeneratePreviewService,
        ValidationService,
        ObsidianScanService,
        TagAuditService,
    );
}

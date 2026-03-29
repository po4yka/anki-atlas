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
pub use validation::{ValidationService, ValidationSummary};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_public_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GeneratePreview>();
        assert_send_sync::<ValidationSummary>();
        assert_send_sync::<ObsidianNotePreview>();
        assert_send_sync::<ObsidianScanPreview>();
        assert_send_sync::<TagAuditEntry>();
        assert_send_sync::<TagAuditSummary>();
        assert_send_sync::<SyncExecutionSummary>();
        assert_send_sync::<IndexExecutionSummary>();
        assert_send_sync::<SyncStatsSummary>();
        assert_send_sync::<GeneratePreviewService>();
        assert_send_sync::<ValidationService>();
        assert_send_sync::<ObsidianScanService>();
        assert_send_sync::<TagAuditService>();
    }
}

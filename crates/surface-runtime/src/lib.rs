mod contracts;
pub mod error;
pub mod services;
pub mod workflows;

pub use error::SurfaceError;
pub use services::{
    AnalyticsFacade, BuildSurfaceServicesOptions, SearchFacade, SurfaceServices,
    build_surface_services,
};
pub use workflows::{
    GeneratePreview, GeneratePreviewService, IndexExecutionSummary, IndexExecutor, IndexingService,
    ObsidianScanPreview, ObsidianScanService, SurfaceOperation, SurfaceProgressEvent,
    SurfaceProgressSink, SyncExecutionHandle, SyncExecutionService, SyncExecutionSummary,
    TagAuditEntry, TagAuditService, TagAuditSummary, ValidationService, ValidationSummary,
};

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
    ObsidianScanPreview, ObsidianScanService, SyncExecutionService, SyncExecutionSummary,
    SyncExecutor, TagAuditEntry, TagAuditService, TagAuditSummary, ValidationService,
    ValidationSummary,
};

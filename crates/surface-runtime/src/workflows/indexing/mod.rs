mod chunk;
mod media;
mod prepare;
mod service;

pub use service::IndexingService;

use indexer::service::IndexStats;
use serde::Serialize;

use crate::error::SurfaceError;
use crate::workflows::progress::SurfaceProgressSink;

#[derive(Debug, Clone, Serialize)]
pub struct IndexExecutionSummary {
    pub force_reindex: bool,
    pub stats: IndexStats,
}

#[async_trait::async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait IndexExecutor: Send + Sync {
    async fn index_all_notes(
        &self,
        force_reindex: bool,
    ) -> Result<IndexExecutionSummary, SurfaceError>;

    async fn index_all_notes_with_progress(
        &self,
        force_reindex: bool,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<IndexExecutionSummary, SurfaceError>;
}

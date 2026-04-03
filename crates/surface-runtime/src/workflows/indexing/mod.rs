mod chunk;
mod media;
mod prepare;
mod service;

pub use service::IndexingService;

use common::ReindexMode;
use indexer::service::IndexStats;
use serde::Serialize;

use crate::error::SurfaceError;
use crate::workflows::progress::SurfaceProgressSink;

#[derive(Debug, Clone, Serialize)]
pub struct IndexExecutionSummary {
    pub reindex_mode: ReindexMode,
    pub stats: IndexStats,
}

#[async_trait::async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait IndexExecutor: Send + Sync {
    async fn index_all_notes(
        &self,
        reindex_mode: ReindexMode,
    ) -> Result<IndexExecutionSummary, SurfaceError>;

    async fn index_all_notes_with_progress(
        &self,
        reindex_mode: ReindexMode,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<IndexExecutionSummary, SurfaceError>;
}

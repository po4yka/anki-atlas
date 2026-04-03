use std::path::PathBuf;
use std::sync::Arc;

use anki_sync::{
    SyncProgressCallback, SyncProgressEvent, SyncProgressStage, SyncStats,
    sync_anki_collection_owned_with_progress,
};
use common::ReindexMode;
use serde::Serialize;
use sqlx::PgPool;

use super::indexing::{IndexExecutionSummary, IndexExecutor, IndexingService};
use super::progress::{SurfaceOperation, SurfaceProgressSink, emit_progress, map_sync_progress};
use crate::error::SurfaceError;

#[derive(Debug, Clone, Serialize)]
pub struct SyncExecutionSummary {
    pub source: PathBuf,
    pub migrations_applied: bool,
    pub sync: SyncStatsSummary,
    pub index: Option<IndexExecutionSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SyncStatsSummary {
    pub decks_upserted: i32,
    pub models_upserted: i32,
    pub notes_upserted: i32,
    pub notes_deleted: i32,
    pub cards_upserted: i32,
    pub card_stats_upserted: i32,
    pub duration_ms: i64,
}

impl From<SyncStats> for SyncStatsSummary {
    fn from(stats: SyncStats) -> Self {
        Self {
            decks_upserted: stats.decks_upserted,
            models_upserted: stats.models_upserted,
            notes_upserted: stats.notes_upserted,
            notes_deleted: stats.notes_deleted,
            cards_upserted: stats.cards_upserted,
            card_stats_upserted: stats.card_stats_upserted,
            duration_ms: stats.duration_ms,
        }
    }
}

pub struct SyncExecutionService {
    db: PgPool,
    indexer: Arc<dyn IndexExecutor>,
}

#[derive(Clone)]
pub struct SyncExecutionHandle {
    db: PgPool,
    indexer: Arc<dyn IndexExecutor>,
}

impl SyncExecutionService {
    pub fn unsupported(db: PgPool) -> Self {
        Self {
            indexer: Arc::new(IndexingService::unsupported(db.clone())),
            db,
        }
    }

    pub fn new(db: PgPool, indexer: Arc<dyn IndexExecutor>) -> Self {
        Self { db, indexer }
    }

    pub fn handle(&self) -> SyncExecutionHandle {
        SyncExecutionHandle {
            db: self.db.clone(),
            indexer: self.indexer.clone(),
        }
    }

    pub async fn sync_collection(
        &self,
        source: PathBuf,
        run_migrations: bool,
        run_index: bool,
        reindex_mode: ReindexMode,
    ) -> Result<SyncExecutionSummary, SurfaceError> {
        self.sync_collection_with_progress(source, run_migrations, run_index, reindex_mode, None)
            .await
    }

    pub async fn sync_collection_with_progress(
        &self,
        source: PathBuf,
        run_migrations: bool,
        run_index: bool,
        reindex_mode: ReindexMode,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<SyncExecutionSummary, SurfaceError> {
        run_sync_collection(
            self.db.clone(),
            self.indexer.clone(),
            source,
            run_migrations,
            run_index,
            reindex_mode,
            progress,
        )
        .await
    }
}

impl SyncExecutionHandle {
    pub async fn sync_collection(
        self,
        source: PathBuf,
        run_migrations: bool,
        run_index: bool,
        reindex_mode: ReindexMode,
    ) -> Result<SyncExecutionSummary, SurfaceError> {
        self.sync_collection_with_progress(source, run_migrations, run_index, reindex_mode, None)
            .await
    }

    pub async fn sync_collection_with_progress(
        self,
        source: PathBuf,
        run_migrations: bool,
        run_index: bool,
        reindex_mode: ReindexMode,
        progress: Option<SurfaceProgressSink>,
    ) -> Result<SyncExecutionSummary, SurfaceError> {
        run_sync_collection(
            self.db,
            self.indexer,
            source,
            run_migrations,
            run_index,
            reindex_mode,
            progress,
        )
        .await
    }
}

async fn run_sync_collection(
    db: PgPool,
    indexer: Arc<dyn IndexExecutor>,
    source: PathBuf,
    run_migrations: bool,
    run_index: bool,
    reindex_mode: ReindexMode,
    progress: Option<SurfaceProgressSink>,
) -> Result<SyncExecutionSummary, SurfaceError> {
    if !source.exists() {
        return Err(SurfaceError::PathNotFound(source));
    }
    if run_migrations {
        emit_progress(
            progress.as_ref(),
            SurfaceOperation::Sync,
            "running_migrations",
            0,
            1,
            "running database migrations",
        );
        database::run_migrations_owned(db.clone()).await?;
        emit_progress(
            progress.as_ref(),
            SurfaceOperation::Sync,
            "running_migrations",
            1,
            1,
            "database migrations complete",
        );
    }
    let mapped_progress = progress.as_ref().map(|sink| {
        let sink = Arc::clone(sink);
        Arc::new(move |event: SyncProgressEvent| {
            sink(map_sync_progress(&event));
        }) as SyncProgressCallback
    });
    let sync =
        sync_anki_collection_owned_with_progress(db, source.clone(), mapped_progress).await?;
    let index = if run_index {
        Some(
            indexer
                .index_all_notes_with_progress(reindex_mode, progress.clone())
                .await?,
        )
    } else {
        None
    };
    emit_progress(
        progress.as_ref(),
        SurfaceOperation::Sync,
        SyncProgressStage::Completed.as_str(),
        1,
        1,
        format!("sync pipeline completed for {}", source.display()),
    );

    Ok(SyncExecutionSummary {
        source,
        migrations_applied: run_migrations,
        sync: sync.into(),
        index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_stats_summary_from_maps_all_fields() {
        let stats = SyncStats {
            decks_upserted: 1,
            models_upserted: 2,
            notes_upserted: 3,
            notes_deleted: 4,
            cards_upserted: 5,
            card_stats_upserted: 6,
            duration_ms: 7,
        };
        let summary = SyncStatsSummary::from(stats);
        assert_eq!(summary.decks_upserted, 1);
        assert_eq!(summary.models_upserted, 2);
        assert_eq!(summary.notes_upserted, 3);
        assert_eq!(summary.notes_deleted, 4);
        assert_eq!(summary.cards_upserted, 5);
        assert_eq!(summary.card_stats_upserted, 6);
        assert_eq!(summary.duration_ms, 7);
    }
}

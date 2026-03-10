use std::path::PathBuf;

use jobs::error::JobError;
use jobs::tasks::TaskContext;
use jobs::types::{IndexJobPayload, IndexJobResult, SyncJobPayload, SyncJobResult};
use surface_runtime::BuildSurfaceServicesOptions;

fn map_task_error(error: impl std::fmt::Display) -> JobError {
    JobError::TaskExecution(error.to_string())
}

async fn build_services() -> Result<surface_runtime::SurfaceServices, JobError> {
    let settings = common::config::Settings::load().map_err(map_task_error)?;
    surface_runtime::build_surface_services(
        &settings,
        BuildSurfaceServicesOptions {
            enable_direct_execution: true,
        },
    )
    .await
    .map_err(map_task_error)
}

pub async fn job_sync(
    _ctx: &TaskContext,
    _job_id: &str,
    payload: &SyncJobPayload,
) -> Result<SyncJobResult, JobError> {
    let services = build_services().await?;
    let summary = services
        .sync
        .sync_collection(
            PathBuf::from(&payload.source),
            payload.run_migrations,
            payload.index,
            payload.force_reindex,
        )
        .await
        .map_err(map_task_error)?;

    let index = summary.index.map(|index| index.stats);

    Ok(SyncJobResult {
        decks_upserted: i64::from(summary.sync.decks_upserted),
        models_upserted: i64::from(summary.sync.models_upserted),
        notes_upserted: i64::from(summary.sync.notes_upserted),
        notes_deleted: i64::from(summary.sync.notes_deleted),
        cards_upserted: i64::from(summary.sync.cards_upserted),
        card_stats_upserted: i64::from(summary.sync.card_stats_upserted),
        duration_ms: summary.sync.duration_ms,
        notes_embedded: index.as_ref().map(|stats| stats.notes_embedded as i64),
        notes_skipped: index.as_ref().map(|stats| stats.notes_skipped as i64),
        index_errors: index.map_or_else(Vec::new, |stats| stats.errors),
    })
}

pub async fn job_index(
    _ctx: &TaskContext,
    _job_id: &str,
    payload: &IndexJobPayload,
) -> Result<IndexJobResult, JobError> {
    let services = build_services().await?;
    let summary = services
        .index
        .index_all_notes(payload.force_reindex)
        .await
        .map_err(map_task_error)?;

    Ok(IndexJobResult {
        notes_processed: summary.stats.notes_processed as i64,
        notes_embedded: summary.stats.notes_embedded as i64,
        notes_skipped: summary.stats.notes_skipped as i64,
        notes_deleted: summary.stats.notes_deleted as i64,
        errors: summary.stats.errors,
    })
}

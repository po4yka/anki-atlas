use std::sync::Arc;

use anki_sync::SyncProgressEvent;
use indexer::progress::{IndexProgressEvent, IndexProgressStage};
use serde::Serialize;

pub type SurfaceProgressSink = Arc<dyn Fn(SurfaceProgressEvent) + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SurfaceOperation {
    Sync,
    Index,
    ObsidianScan,
}

#[derive(Debug, Clone, Serialize)]
pub struct SurfaceProgressEvent {
    pub operation: SurfaceOperation,
    pub stage: String,
    pub current: usize,
    pub total: usize,
    pub message: String,
}

pub(super) fn emit_progress(
    sink: Option<&SurfaceProgressSink>,
    operation: SurfaceOperation,
    stage: impl Into<String>,
    current: usize,
    total: usize,
    message: impl Into<String>,
) {
    if let Some(sink) = sink {
        sink(SurfaceProgressEvent {
            operation,
            stage: stage.into(),
            current,
            total,
            message: message.into(),
        });
    }
}

pub(super) fn map_sync_progress(progress: &SyncProgressEvent) -> SurfaceProgressEvent {
    SurfaceProgressEvent {
        operation: SurfaceOperation::Sync,
        stage: progress.stage.as_str().to_string(),
        current: progress.current,
        total: progress.total,
        message: progress.message.clone(),
    }
}

pub(super) fn map_index_progress(progress: &IndexProgressEvent) -> Option<SurfaceProgressEvent> {
    if progress.stage == IndexProgressStage::Completed {
        return None;
    }

    Some(SurfaceProgressEvent {
        operation: SurfaceOperation::Index,
        stage: progress.stage.as_str().to_string(),
        current: progress.current,
        total: progress.total,
        message: progress.message.clone(),
    })
}

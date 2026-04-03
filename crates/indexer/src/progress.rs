use std::sync::Arc;

/// Progress callback for indexing work.
pub type IndexProgressCallback = Arc<dyn Fn(IndexProgressEvent) + Send + Sync>;

/// Stage-level progress emitted during indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexProgressStage {
    Starting,
    HashLookup,
    Diffing,
    Embedding,
    Upserting,
    Completed,
}

impl IndexProgressStage {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::HashLookup => "hash_lookup",
            Self::Diffing => "diffing",
            Self::Embedding => "embedding",
            Self::Upserting => "upserting",
            Self::Completed => "completed",
        }
    }
}

/// Progress event emitted during indexing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexProgressEvent {
    pub stage: IndexProgressStage,
    pub current: usize,
    pub total: usize,
    pub message: String,
}

pub(crate) fn emit_progress(
    callback: Option<&IndexProgressCallback>,
    stage: IndexProgressStage,
    current: usize,
    total: usize,
    message: impl Into<String>,
) {
    if let Some(callback) = callback {
        callback(IndexProgressEvent {
            stage,
            current,
            total,
            message: message.into(),
        });
    }
}

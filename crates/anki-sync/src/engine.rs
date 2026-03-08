use crate::progress::ProgressTracker;
use crate::state::StateDB;

/// Result of an engine-level sync operation.
#[derive(Debug, Clone)]
pub struct SyncResult {
    pub cards_created: i32,
    pub cards_updated: i32,
    pub cards_deleted: i32,
    pub cards_skipped: i32,
    pub errors: i32,
    pub duration_ms: i64,
}

/// Phased sync engine with pluggable state and progress tracking.
///
/// Lifecycle: INITIALIZING -> SCANNING -> APPLYING -> COMPLETED/FAILED
pub struct SyncEngine {
    _state_db: StateDB,
    _progress: ProgressTracker,
}

impl SyncEngine {
    pub fn new(_state_db: StateDB, _progress: Option<ProgressTracker>) -> Self {
        todo!()
    }

    /// Access the state database.
    pub fn state_db(&self) -> &StateDB {
        todo!()
    }

    /// Access the progress tracker.
    pub fn progress(&self) -> &ProgressTracker {
        todo!()
    }

    /// Run the sync lifecycle.
    pub fn sync(&mut self, _dry_run: bool) -> Result<SyncResult, Box<dyn std::error::Error>> {
        todo!()
    }
}

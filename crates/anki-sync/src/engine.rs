use std::time::Instant;

use crate::progress::{ProgressTracker, SyncPhase};
use crate::state::{StateDB, StateDbError};

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
    state_db: StateDB,
    progress: ProgressTracker,
}

impl SyncEngine {
    pub fn new(state_db: StateDB, progress: Option<ProgressTracker>) -> Self {
        let progress = progress.unwrap_or_else(|| ProgressTracker::new(None));
        Self { state_db, progress }
    }

    /// Access the state database.
    pub fn state_db(&self) -> &StateDB {
        &self.state_db
    }

    /// Access the progress tracker.
    pub fn progress(&self) -> &ProgressTracker {
        &self.progress
    }

    /// Run the sync lifecycle.
    pub fn sync(&mut self, dry_run: bool) -> Result<SyncResult, StateDbError> {
        let start = Instant::now();

        // SCANNING phase
        self.progress.set_phase(SyncPhase::Scanning);
        let states = self.state_db.get_all()?;
        self.progress.set_total(states.len() as i32);

        // APPLYING phase (skip for dry run)
        if !dry_run {
            self.progress.set_phase(SyncPhase::Applying);
        }

        // COMPLETED
        self.progress.set_phase(SyncPhase::Completed);

        let duration_ms = start.elapsed().as_millis() as i64;

        Ok(SyncResult {
            cards_created: 0,
            cards_updated: 0,
            cards_deleted: 0,
            cards_skipped: 0,
            errors: 0,
            duration_ms,
        })
    }
}

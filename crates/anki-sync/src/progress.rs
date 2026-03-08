/// Phases of a sync operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPhase {
    Initializing,
    Indexing,
    Scanning,
    Generating,
    Applying,
    Completed,
    Failed,
}

impl SyncPhase {
    pub fn as_str(&self) -> &'static str {
        todo!()
    }
}

/// Frozen snapshot of sync progress.
#[derive(Debug, Clone)]
pub struct SyncProgress {
    pub session_id: String,
    pub phase: SyncPhase,
    pub total_notes: i32,
    pub notes_processed: i32,
    pub cards_created: i32,
    pub cards_updated: i32,
    pub cards_deleted: i32,
    pub errors: i32,
    pub started_at: f64,
    pub updated_at: f64,
}

/// Valid stat names for `increment`.
pub const VALID_STATS: &[&str] = &[
    "notes_processed",
    "cards_created",
    "cards_updated",
    "cards_deleted",
    "errors",
];

/// Thread-safe progress tracker.
pub struct ProgressTracker {
    _private: (),
}

impl ProgressTracker {
    /// Create a new tracker with a random or provided session ID.
    pub fn new(_session_id: Option<String>) -> Self {
        todo!()
    }

    /// Set the current phase.
    pub fn set_phase(&self, _phase: SyncPhase) {
        todo!()
    }

    /// Set the total number of notes to process.
    pub fn set_total(&self, _total: i32) {
        todo!()
    }

    /// Increment a named stat by `count`.
    /// Panics if `stat` is not in VALID_STATS.
    pub fn increment(&self, _stat: &str, _count: i32) {
        todo!()
    }

    /// Get a frozen snapshot of current progress.
    pub fn snapshot(&self) -> SyncProgress {
        todo!()
    }

    /// Mark sync as completed (success=true) or failed (success=false).
    pub fn complete(&self, _success: bool) {
        todo!()
    }

    /// Get progress percentage (0.0 to 100.0).
    pub fn progress_pct(&self) -> f64 {
        todo!()
    }
}

impl Clone for ProgressTracker {
    fn clone(&self) -> Self {
        todo!()
    }
}

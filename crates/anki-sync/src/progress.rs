use std::sync::{Arc, Mutex};

use crate::now_secs;

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
        match self {
            Self::Initializing => "initializing",
            Self::Indexing => "indexing",
            Self::Scanning => "scanning",
            Self::Generating => "generating",
            Self::Applying => "applying",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
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

struct ProgressState {
    session_id: String,
    phase: SyncPhase,
    total_notes: i32,
    notes_processed: i32,
    cards_created: i32,
    cards_updated: i32,
    cards_deleted: i32,
    errors: i32,
    started_at: f64,
    updated_at: f64,
}

/// Thread-safe progress tracker.
///
/// All methods acquire an internal `Mutex` before reading/writing.
pub struct ProgressTracker {
    inner: Arc<Mutex<ProgressState>>,
}

impl ProgressTracker {
    /// Create a new tracker with a random or provided session ID.
    pub fn new(session_id: Option<String>) -> Self {
        let sid = session_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let now = now_secs();
        Self {
            inner: Arc::new(Mutex::new(ProgressState {
                session_id: sid,
                phase: SyncPhase::Initializing,
                total_notes: 0,
                notes_processed: 0,
                cards_created: 0,
                cards_updated: 0,
                cards_deleted: 0,
                errors: 0,
                started_at: now,
                updated_at: now,
            })),
        }
    }

    /// Set the current phase.
    pub fn set_phase(&self, phase: SyncPhase) {
        let mut state = self.inner.lock().expect("poisoned lock");
        state.phase = phase;
        state.updated_at = now_secs();
    }

    /// Set the total number of notes to process.
    pub fn set_total(&self, total: i32) {
        let mut state = self.inner.lock().expect("poisoned lock");
        state.total_notes = total;
        state.updated_at = now_secs();
    }

    /// Increment a named stat by `count`.
    /// Panics if `stat` is not in VALID_STATS.
    pub fn increment(&self, stat: &str, count: i32) {
        let mut state = self.inner.lock().expect("poisoned lock");
        match stat {
            "notes_processed" => state.notes_processed += count,
            "cards_created" => state.cards_created += count,
            "cards_updated" => state.cards_updated += count,
            "cards_deleted" => state.cards_deleted += count,
            "errors" => state.errors += count,
            _ => panic!("invalid stat name: {stat}"),
        }
        state.updated_at = now_secs();
    }

    /// Get a frozen snapshot of current progress.
    pub fn snapshot(&self) -> SyncProgress {
        let state = self.inner.lock().expect("poisoned lock");
        SyncProgress {
            session_id: state.session_id.clone(),
            phase: state.phase,
            total_notes: state.total_notes,
            notes_processed: state.notes_processed,
            cards_created: state.cards_created,
            cards_updated: state.cards_updated,
            cards_deleted: state.cards_deleted,
            errors: state.errors,
            started_at: state.started_at,
            updated_at: state.updated_at,
        }
    }

    /// Mark sync as completed (success=true) or failed (success=false).
    pub fn complete(&self, success: bool) {
        let phase = if success {
            SyncPhase::Completed
        } else {
            SyncPhase::Failed
        };
        self.set_phase(phase);
    }

    /// Get progress percentage (0.0 to 100.0).
    pub fn progress_pct(&self) -> f64 {
        let state = self.inner.lock().expect("poisoned lock");
        if state.total_notes == 0 {
            return 0.0;
        }
        let pct = (state.notes_processed as f64 / state.total_notes as f64) * 100.0;
        pct.min(100.0)
    }
}

impl Clone for ProgressTracker {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

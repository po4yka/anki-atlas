pub mod core;
pub mod engine;
pub mod progress;
pub mod recovery;
pub mod state;

use std::time::{SystemTime, UNIX_EPOCH};

/// Current time as seconds since Unix epoch.
pub(crate) fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time went backwards")
        .as_secs_f64()
}

pub use core::{
    SyncProgressCallback, SyncProgressEvent, SyncProgressStage, SyncService, SyncStats,
    sync_anki_collection, sync_anki_collection_owned, sync_anki_collection_owned_with_progress,
};
pub use engine::{SyncEngine, SyncResult};
pub use progress::{ProgressTracker, SyncPhase, SyncProgress, VALID_STATS};
pub use recovery::{CardRecovery, CardTransaction, RollbackAction};
pub use state::{CardState, StateDB, StateDbError};

#[cfg(test)]
mod send_sync_tests {
    common::assert_send_sync!(
        super::CardState,
        super::SyncPhase,
        super::SyncProgress,
        super::ProgressTracker,
        super::SyncResult,
        super::SyncProgressEvent,
        super::SyncProgressCallback,
        super::CardTransaction,
        super::RollbackAction,
        super::SyncStats,
        super::SyncService,
        super::StateDB,
        super::SyncEngine,
    );
}

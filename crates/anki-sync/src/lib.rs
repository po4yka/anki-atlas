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

pub use core::{SyncService, SyncStats, sync_anki_collection};
pub use engine::{SyncEngine, SyncResult};
pub use progress::{ProgressTracker, SyncPhase, SyncProgress, VALID_STATS};
pub use recovery::{CardRecovery, CardTransaction, RollbackAction};
pub use state::{CardState, StateDB, StateDbError};

#[cfg(test)]
mod send_sync_tests {
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn public_types_are_send_sync() {
        assert_send_sync::<super::CardState>();
        assert_send_sync::<super::SyncPhase>();
        assert_send_sync::<super::SyncProgress>();
        assert_send_sync::<super::ProgressTracker>();
        assert_send_sync::<super::SyncResult>();
        assert_send_sync::<super::CardTransaction>();
        assert_send_sync::<super::RollbackAction>();
        assert_send_sync::<super::SyncStats>();
        assert_send_sync::<super::SyncService>();
        assert_send_sync::<super::StateDB>();
        assert_send_sync::<super::SyncEngine>();
    }
}

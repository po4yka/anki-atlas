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

pub use engine::{SyncEngine, SyncResult};
pub use progress::{ProgressTracker, SyncPhase, SyncProgress, VALID_STATS};
pub use recovery::{CardRecovery, CardTransaction, RollbackAction};
pub use state::{CardState, StateDB};

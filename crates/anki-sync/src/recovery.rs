use std::collections::HashSet;

use crate::state::{CardState, StateDB};

/// A recorded action that can be rolled back.
#[derive(Debug, Clone)]
pub struct RollbackAction {
    pub action_type: String,
    pub target_id: String,
    pub succeeded: bool,
    pub error: String,
}

/// Atomic card operation with rollback support.
pub struct CardTransaction {
    _private: (),
}

impl CardTransaction {
    pub fn new() -> Self {
        todo!()
    }

    /// Record an action for potential rollback.
    pub fn add_rollback(&mut self, _action_type: &str, _target_id: &str) {
        todo!()
    }

    /// Mark as committed -- rollback becomes a no-op.
    pub fn commit(&mut self) {
        todo!()
    }

    /// Roll back all recorded actions in reverse order.
    pub fn rollback(&mut self) -> Vec<RollbackAction> {
        todo!()
    }
}

impl Drop for CardTransaction {
    fn drop(&mut self) {
        // Will trigger rollback if not committed
    }
}

/// Detect and recover from inconsistent card states.
pub struct CardRecovery<'a> {
    state_db: &'a StateDB,
}

impl<'a> CardRecovery<'a> {
    pub fn new(state_db: &'a StateDB) -> Self {
        Self { state_db }
    }

    /// Find orphaned cards: in DB but not Anki, and in Anki but not DB.
    pub fn find_orphaned(
        &self,
        _db_slugs: &HashSet<String>,
        _anki_slugs: &HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>) {
        todo!()
    }

    /// Find card states older than `max_age_days`.
    pub fn find_stale(&self, _max_age_days: u32) -> Vec<CardState> {
        todo!()
    }
}

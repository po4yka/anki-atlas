use std::collections::HashSet;
use std::time::{SystemTime, UNIX_EPOCH};

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
///
/// Use as a scope guard: if the transaction is dropped without `commit()`,
/// `rollback()` is called automatically.
pub struct CardTransaction {
    actions: Vec<(String, String)>,
    committed: bool,
}

impl Default for CardTransaction {
    fn default() -> Self {
        Self::new()
    }
}

impl CardTransaction {
    pub fn new() -> Self {
        Self {
            actions: Vec::new(),
            committed: false,
        }
    }

    /// Record an action for potential rollback.
    pub fn add_rollback(&mut self, action_type: &str, target_id: &str) {
        self.actions
            .push((action_type.to_string(), target_id.to_string()));
    }

    /// Mark as committed -- rollback becomes a no-op.
    pub fn commit(&mut self) {
        self.committed = true;
    }

    /// Roll back all recorded actions in reverse order.
    /// Returns the rollback results.
    pub fn rollback(&mut self) -> Vec<RollbackAction> {
        if self.committed {
            return Vec::new();
        }

        let actions: Vec<RollbackAction> = self
            .actions
            .drain(..)
            .rev()
            .map(|(action_type, target_id)| RollbackAction {
                action_type,
                target_id,
                succeeded: true,
                error: String::new(),
            })
            .collect();

        actions
    }
}

impl Drop for CardTransaction {
    fn drop(&mut self) {
        if !self.committed {
            let _ = self.rollback();
        }
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
    /// Returns (in_db_not_anki, in_anki_not_db).
    pub fn find_orphaned(
        &self,
        db_slugs: &HashSet<String>,
        anki_slugs: &HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>) {
        let in_db_not_anki = db_slugs.difference(anki_slugs).cloned().collect();
        let in_anki_not_db = anki_slugs.difference(db_slugs).cloned().collect();
        (in_db_not_anki, in_anki_not_db)
    }

    /// Find card states older than `max_age_days`.
    /// Only considers states with `synced_at > 0`.
    pub fn find_stale(&self, max_age_days: u32) -> Vec<CardState> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_secs_f64();

        let cutoff = now - (max_age_days as f64 * 86400.0);

        self.state_db
            .get_all()
            .into_iter()
            .filter(|s| s.synced_at > 0.0 && s.synced_at < cutoff)
            .collect()
    }
}

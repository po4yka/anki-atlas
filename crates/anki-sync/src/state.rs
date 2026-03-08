use std::path::Path;

/// Tracked state for a single card.
#[derive(Debug, Clone, PartialEq)]
pub struct CardState {
    pub slug: String,
    pub content_hash: String,
    pub anki_guid: Option<i64>,
    pub note_type: String,
    pub source_path: String,
    pub synced_at: f64,
}

/// SQLite WAL database for tracking per-card sync state.
pub struct StateDB {
    _private: (),
}

impl StateDB {
    /// Open or create the state database at `db_path`.
    pub fn open(_db_path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        todo!()
    }

    /// Get card state by slug, or `None` if not found.
    pub fn get(&self, _slug: &str) -> Option<CardState> {
        todo!()
    }

    /// Get all card states, sorted by slug.
    pub fn get_all(&self) -> Vec<CardState> {
        todo!()
    }

    /// Insert or update a card state (upsert on slug).
    pub fn upsert(&self, _state: &CardState) {
        todo!()
    }

    /// Delete card state by slug.
    pub fn delete(&self, _slug: &str) {
        todo!()
    }

    /// Get all card states for a given source path, sorted by slug.
    pub fn get_by_source(&self, _source_path: &str) -> Vec<CardState> {
        todo!()
    }

    /// Close the database connection.
    pub fn close(self) {
        todo!()
    }
}

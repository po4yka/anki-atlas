use std::path::Path;
use std::sync::Mutex;

use rusqlite::{Connection, Row};

/// Errors from StateDB operations.
#[derive(Debug, thiserror::Error)]
pub enum StateDbError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("mutex poisoned")]
    MutexPoisoned,
}

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

impl CardState {
    /// Map a SQLite row to a `CardState`.
    ///
    /// Expects columns in order: slug, content_hash, anki_guid, note_type, source_path, synced_at.
    fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            slug: row.get(0)?,
            content_hash: row.get(1)?,
            anki_guid: row.get(2)?,
            note_type: row.get(3)?,
            source_path: row.get(4)?,
            synced_at: row.get(5)?,
        })
    }
}

/// SQLite WAL database for tracking per-card sync state.
pub struct StateDB {
    conn: Mutex<Connection>,
}

impl StateDB {
    /// Open or create the state database at `db_path`.
    /// Enables WAL mode and foreign keys. Creates the `card_state` table if needed.
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self, StateDbError> {
        let conn = Connection::open(db_path.as_ref())?;

        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS card_state (
                slug TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                anki_guid INTEGER,
                note_type TEXT NOT NULL DEFAULT '',
                source_path TEXT NOT NULL DEFAULT '',
                synced_at REAL NOT NULL DEFAULT 0.0
            );",
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Get card state by slug, or `None` if not found.
    pub fn get(&self, slug: &str) -> Result<Option<CardState>, StateDbError> {
        let conn = self.conn.lock().map_err(|_| StateDbError::MutexPoisoned)?;
        match conn.query_row(
            "SELECT slug, content_hash, anki_guid, note_type, source_path, synced_at
                 FROM card_state WHERE slug = ?1",
            [slug],
            CardState::from_row,
        ) {
            Ok(state) => Ok(Some(state)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(StateDbError::Sqlite(e)),
        }
    }

    /// Get all card states, sorted by slug.
    pub fn get_all(&self) -> Result<Vec<CardState>, StateDbError> {
        let conn = self.conn.lock().map_err(|_| StateDbError::MutexPoisoned)?;
        let mut stmt = conn.prepare(
            "SELECT slug, content_hash, anki_guid, note_type, source_path, synced_at
                 FROM card_state ORDER BY slug",
        )?;

        let rows = stmt.query_map([], CardState::from_row)?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Insert or update a card state (upsert on slug).
    pub fn upsert(&self, state: &CardState) -> Result<(), StateDbError> {
        let conn = self.conn.lock().map_err(|_| StateDbError::MutexPoisoned)?;
        conn.execute(
            "INSERT INTO card_state (slug, content_hash, anki_guid, note_type, source_path, synced_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(slug) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    anki_guid = excluded.anki_guid,
                    note_type = excluded.note_type,
                    source_path = excluded.source_path,
                    synced_at = excluded.synced_at",
            rusqlite::params![
                state.slug,
                state.content_hash,
                state.anki_guid,
                state.note_type,
                state.source_path,
                state.synced_at,
            ],
        )?;
        Ok(())
    }

    /// Delete card state by slug.
    pub fn delete(&self, slug: &str) -> Result<(), StateDbError> {
        let conn = self.conn.lock().map_err(|_| StateDbError::MutexPoisoned)?;
        conn.execute("DELETE FROM card_state WHERE slug = ?1", [slug])?;
        Ok(())
    }

    /// Get all card states for a given source path, sorted by slug.
    pub fn get_by_source(&self, source_path: &str) -> Result<Vec<CardState>, StateDbError> {
        let conn = self.conn.lock().map_err(|_| StateDbError::MutexPoisoned)?;
        let mut stmt = conn.prepare(
            "SELECT slug, content_hash, anki_guid, note_type, source_path, synced_at
                 FROM card_state WHERE source_path = ?1 ORDER BY slug",
        )?;

        let rows = stmt.query_map([source_path], CardState::from_row)?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Close the database connection.
    pub fn close(self) {
        // Connection is closed when dropped
    }
}

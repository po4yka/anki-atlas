use std::path::{Path, PathBuf};
use std::sync::Mutex;

use rusqlite::{Connection, Row};

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
    #[allow(dead_code)]
    path: PathBuf,
}

impl StateDB {
    /// Open or create the state database at `db_path`.
    /// Enables WAL mode and foreign keys. Creates the `card_state` table if needed.
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let path = db_path.as_ref().to_path_buf();
        let conn = Connection::open(&path)?;

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
            path,
        })
    }

    /// Get card state by slug, or `None` if not found.
    pub fn get(&self, slug: &str) -> Option<CardState> {
        let conn = self.conn.lock().expect("state_db mutex poisoned");
        conn.query_row(
            "SELECT slug, content_hash, anki_guid, note_type, source_path, synced_at
                 FROM card_state WHERE slug = ?1",
            [slug],
            CardState::from_row,
        )
        .ok()
    }

    /// Get all card states, sorted by slug.
    pub fn get_all(&self) -> Vec<CardState> {
        let conn = self.conn.lock().expect("state_db mutex poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT slug, content_hash, anki_guid, note_type, source_path, synced_at
                 FROM card_state ORDER BY slug",
            )
            .expect("failed to prepare get_all statement");

        stmt.query_map([], CardState::from_row)
            .expect("failed to query card_state")
            .filter_map(Result::ok)
            .collect()
    }

    /// Insert or update a card state (upsert on slug).
    pub fn upsert(&self, state: &CardState) {
        let conn = self.conn.lock().expect("state_db mutex poisoned");
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
        )
        .expect("failed to upsert card_state");
    }

    /// Delete card state by slug.
    pub fn delete(&self, slug: &str) {
        let conn = self.conn.lock().expect("state_db mutex poisoned");
        conn.execute("DELETE FROM card_state WHERE slug = ?1", [slug])
            .expect("failed to delete card_state");
    }

    /// Get all card states for a given source path, sorted by slug.
    pub fn get_by_source(&self, source_path: &str) -> Vec<CardState> {
        let conn = self.conn.lock().expect("state_db mutex poisoned");
        let mut stmt = conn
            .prepare(
                "SELECT slug, content_hash, anki_guid, note_type, source_path, synced_at
                 FROM card_state WHERE source_path = ?1 ORDER BY slug",
            )
            .expect("failed to prepare get_by_source statement");

        stmt.query_map([source_path], CardState::from_row)
            .expect("failed to query card_state by source")
            .filter_map(Result::ok)
            .collect()
    }

    /// Close the database connection.
    pub fn close(self) {
        // Connection is closed when dropped
    }
}

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Schema version for migration tracking.
pub const SCHEMA_VERSION: u32 = 2;

/// Registry entry for a tracked card.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CardEntry {
    pub slug: String,
    pub note_id: String,
    pub source_path: String,
    pub front: String,
    pub back: String,
    pub content_hash: String,
    pub metadata_hash: String,
    pub language: String,
    pub tags: Vec<String>,
    pub anki_note_id: Option<i64>,
    pub created_at: Option<DateTime<Utc>>,
    pub updated_at: Option<DateTime<Utc>>,
    pub synced_at: Option<DateTime<Utc>>,
}

/// Registry entry for a tracked note.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NoteEntry {
    pub note_id: String,
    pub source_path: String,
    pub title: Option<String>,
    pub content_hash: Option<String>,
    pub created_at: Option<DateTime<Utc>>,
    pub updated_at: Option<DateTime<Utc>>,
}

/// Registry errors.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("migration failed from v{from} to v{to}: {reason}")]
    Migration { from: u32, to: u32, reason: String },
    #[error("duplicate slug: {0}")]
    DuplicateSlug(String),
}

/// SQLite-backed card registry with automatic schema migrations.
pub struct CardRegistry {
    conn: rusqlite::Connection,
}

impl CardRegistry {
    /// Open or create a registry at the given path. Use ":memory:" for tests.
    /// Runs schema migrations automatically.
    pub fn open(db_path: &str) -> Result<Self, RegistryError> {
        todo!()
    }

    /// Close the connection.
    pub fn close(self) {
        todo!()
    }

    // --- Card CRUD ---

    /// Insert a card. Returns Ok(true) on success, Ok(false) if slug exists.
    pub fn add_card(&self, entry: &CardEntry) -> Result<bool, RegistryError> {
        todo!()
    }

    /// Get card by slug.
    pub fn get_card(&self, slug: &str) -> Result<Option<CardEntry>, RegistryError> {
        todo!()
    }

    /// Update card by slug. Returns Ok(true) if updated.
    pub fn update_card(&self, entry: &CardEntry) -> Result<bool, RegistryError> {
        todo!()
    }

    /// Delete card by slug. Returns Ok(true) if deleted.
    pub fn delete_card(&self, slug: &str) -> Result<bool, RegistryError> {
        todo!()
    }

    /// Find cards by optional filters (note_id, source_path, content_hash).
    pub fn find_cards(
        &self,
        note_id: Option<&str>,
        source_path: Option<&str>,
        content_hash: Option<&str>,
    ) -> Result<Vec<CardEntry>, RegistryError> {
        todo!()
    }

    // --- Note CRUD ---

    /// Insert a note. Returns Ok(true) on success, Ok(false) if note_id exists.
    pub fn add_note(&self, entry: &NoteEntry) -> Result<bool, RegistryError> {
        todo!()
    }

    /// Get note by note_id.
    pub fn get_note(&self, note_id: &str) -> Result<Option<NoteEntry>, RegistryError> {
        todo!()
    }

    /// List all notes ordered by note_id.
    pub fn list_notes(&self) -> Result<Vec<NoteEntry>, RegistryError> {
        todo!()
    }

    // --- Stats ---

    /// Return total number of cards.
    pub fn card_count(&self) -> Result<usize, RegistryError> {
        todo!()
    }

    /// Return total number of notes.
    pub fn note_count(&self) -> Result<usize, RegistryError> {
        todo!()
    }

    // --- Mapping ---

    /// Get all cards for a given note.
    pub fn get_mapping(&self, note_id: &str) -> Result<Vec<CardEntry>, RegistryError> {
        todo!()
    }

    /// Replace all cards for a note (delete + re-insert).
    pub fn update_mapping(&self, note_id: &str, cards: &[CardEntry]) -> Result<(), RegistryError> {
        todo!()
    }
}

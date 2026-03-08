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

// TODO(ralph): Implement CardRegistry and RegistryError

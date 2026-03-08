use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Lightweight mapping entry for a card.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CardMappingEntry {
    pub slug: String,
    pub language: String,
    pub anki_note_id: Option<i64>,
    pub synced_at: Option<DateTime<Utc>>,
    pub content_hash: String,
}

impl CardMappingEntry {
    /// True if anki_note_id is Some.
    pub fn is_synced(&self) -> bool {
        todo!()
    }

    /// Construct from a CardEntry.
    pub fn from_card_entry(entry: &super::registry::CardEntry) -> Self {
        todo!()
    }
}

/// Mapping of a source note to its generated cards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteMapping {
    pub note_path: String,
    pub note_id: String,
    pub note_title: String,
    pub cards: Vec<CardMappingEntry>,
    pub last_sync: Option<DateTime<Utc>>,
    pub is_orphan: bool,
}

impl NoteMapping {
    pub fn card_count(&self) -> usize {
        todo!()
    }

    pub fn synced_count(&self) -> usize {
        todo!()
    }

    pub fn unsynced_count(&self) -> usize {
        todo!()
    }
}

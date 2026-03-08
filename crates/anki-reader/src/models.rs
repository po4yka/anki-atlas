use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnkiDeck {
    pub deck_id: i64,
    pub name: String,
    pub parent_name: Option<String>,
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnkiModel {
    pub model_id: i64,
    pub name: String,
    pub fields: Vec<serde_json::Value>,
    pub templates: Vec<serde_json::Value>,
    pub config: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnkiNote {
    pub note_id: i64,
    pub model_id: i64,
    pub tags: Vec<String>,
    pub fields: Vec<String>,
    pub fields_json: HashMap<String, String>,
    pub raw_fields: Option<String>,
    pub normalized_text: String,
    pub mtime: i64,
    pub usn: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnkiCard {
    pub card_id: i64,
    pub note_id: i64,
    pub deck_id: i64,
    pub ord: i32,
    pub due: Option<i32>,
    pub ivl: i32,
    pub ease: i32,
    pub lapses: i32,
    pub reps: i32,
    pub queue: i32,
    pub card_type: i32,
    pub mtime: i64,
    pub usn: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnkiRevlogEntry {
    pub id: i64,
    pub card_id: i64,
    pub usn: i32,
    pub button_chosen: i32,
    pub interval: i64,
    pub last_interval: i64,
    pub ease: i32,
    pub time_ms: i64,
    pub review_type: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardStats {
    pub card_id: i64,
    pub reviews: i32,
    pub avg_ease: Option<f64>,
    pub fail_rate: Option<f64>,
    pub last_review_at: Option<DateTime<Utc>>,
    pub total_time_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnkiCollection {
    pub decks: Vec<AnkiDeck>,
    pub models: Vec<AnkiModel>,
    pub notes: Vec<AnkiNote>,
    pub cards: Vec<AnkiCard>,
    pub card_stats: Vec<CardStats>,
    pub collection_path: Option<String>,
    pub extracted_at: DateTime<Utc>,
    pub schema_version: i32,
}

impl Default for AnkiCollection {
    fn default() -> Self {
        Self {
            decks: Vec::new(),
            models: Vec::new(),
            notes: Vec::new(),
            cards: Vec::new(),
            card_stats: Vec::new(),
            collection_path: None,
            extracted_at: Utc::now(),
            schema_version: 0, // Wrong on purpose - should be 11
        }
    }
}

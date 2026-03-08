use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

/// Supported card languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, EnumString, Display)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Language {
    En,
    Ru,
    De,
    Fr,
    Es,
    It,
    Pt,
    Zh,
    Ja,
    Ko,
}

/// A URL-safe slug string (e.g. "kotlin-coroutines-basics").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SlugStr(pub String);

/// Anki card identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CardId(pub i64);

/// Anki note identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NoteId(pub i64);

/// Anki deck name (may contain `::` hierarchy separators).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeckName(pub String);

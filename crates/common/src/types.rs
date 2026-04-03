use std::fmt;

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
#[serde(transparent)]
pub struct SlugStr(pub String);

/// Anki card identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CardId(pub i64);

impl fmt::Display for CardId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i64> for CardId {
    fn from(id: i64) -> Self {
        Self(id)
    }
}

impl From<CardId> for i64 {
    fn from(id: CardId) -> Self {
        id.0
    }
}

/// Anki note identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NoteId(pub i64);

impl fmt::Display for NoteId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i64> for NoteId {
    fn from(id: i64) -> Self {
        Self(id)
    }
}

impl From<NoteId> for i64 {
    fn from(id: NoteId) -> Self {
        id.0
    }
}

/// Anki model (note type) identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub i64);

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i64> for ModelId {
    fn from(id: i64) -> Self {
        Self(id)
    }
}

impl From<ModelId> for i64 {
    fn from(id: ModelId) -> Self {
        id.0
    }
}

/// Anki deck identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DeckId(pub i64);

impl fmt::Display for DeckId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i64> for DeckId {
    fn from(id: i64) -> Self {
        Self(id)
    }
}

impl From<DeckId> for i64 {
    fn from(id: DeckId) -> Self {
        id.0
    }
}

/// Anki deck name (may contain `::` hierarchy separators).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DeckName(pub String);

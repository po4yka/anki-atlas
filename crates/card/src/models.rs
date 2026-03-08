use serde::{Deserialize, Serialize};

/// Valid Anki note types.
pub const VALID_NOTE_TYPES: &[&str] = &["APF::Simple", "APF::Cloze", "Basic", "Cloze"];

/// Sync action type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncActionType {
    Create,
    Update,
    Delete,
    Skip,
}

/// Cognitive load level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CognitiveLoad {
    Basic,
    Intermediate,
    Advanced,
}

/// Validation error for card domain types.
#[derive(Debug)]
pub struct CardValidationError {
    pub messages: Vec<String>,
}

impl std::fmt::Display for CardValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "card validation failed: {:?}", self.messages)
    }
}

impl std::error::Error for CardValidationError {}

/// Value object linking a card to its source Obsidian note.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CardManifest {
    pub slug: String,
    pub slug_base: String,
    pub lang: String,
    pub source_path: String,
    pub source_anchor: String,
    pub note_id: String,
    pub note_title: String,
    pub card_index: u32,
    pub guid: Option<String>,
    pub hash6: Option<String>,
    pub obsidian_uri: Option<String>,
    pub difficulty: Option<f64>,
    pub cognitive_load: Option<CognitiveLoad>,
}

/// Domain entity representing an Anki flashcard. Immutable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Card {
    pub slug: String,
    pub language: String,
    pub apf_html: String,
    pub manifest: CardManifest,
    pub note_type: String,
    pub tags: Vec<String>,
    pub anki_guid: Option<String>,
}

/// A sync action for a card.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyncAction {
    pub action_type: SyncActionType,
    pub card: Card,
    pub anki_guid: Option<String>,
    pub reason: Option<String>,
}

impl CardManifest {
    /// Validate and construct. Returns Err with all validation failures.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        slug: String,
        slug_base: String,
        lang: String,
        source_path: String,
        source_anchor: String,
        note_id: String,
        note_title: String,
        card_index: u32,
        guid: Option<String>,
        hash6: Option<String>,
        obsidian_uri: Option<String>,
        difficulty: Option<f64>,
        cognitive_load: Option<CognitiveLoad>,
    ) -> Result<Self, CardValidationError> {
        todo!("implement CardManifest validation")
    }

    /// Obsidian wikilink: [[folder/note#anchor]].
    pub fn anchor_url(&self) -> String {
        todo!("implement anchor_url")
    }

    /// True if note_id and source_path are non-empty.
    pub fn is_linked_to_note(&self) -> bool {
        todo!("implement is_linked_to_note")
    }

    /// Derive a copy with the given Anki GUID.
    pub fn with_guid(&self, guid: String) -> Self {
        todo!("implement with_guid")
    }

    /// Derive a copy with the given content hash.
    pub fn with_hash(&self, hash6: String) -> Self {
        todo!("implement with_hash")
    }

    /// Derive a copy with the given Obsidian URI.
    pub fn with_obsidian_uri(&self, uri: String) -> Self {
        todo!("implement with_obsidian_uri")
    }

    /// Derive a copy with FSRS metadata.
    pub fn with_fsrs_metadata(&self, difficulty: f64, cognitive_load: CognitiveLoad) -> Self {
        todo!("implement with_fsrs_metadata")
    }
}

impl Card {
    /// Validate and construct.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        slug: String,
        language: String,
        apf_html: String,
        manifest: CardManifest,
        note_type: String,
        tags: Vec<String>,
        anki_guid: Option<String>,
    ) -> Result<Self, CardValidationError> {
        todo!("implement Card validation")
    }

    /// True if anki_guid is None.
    pub fn is_new(&self) -> bool {
        todo!("implement is_new")
    }

    /// SHA-256[:6] of "{apf_html}|{note_type}|{sorted,tags}".
    pub fn content_hash(&self) -> String {
        todo!("implement content_hash")
    }

    /// Derive copy with Anki GUID set on both Card and Manifest.
    pub fn with_guid(&self, guid: String) -> Result<Self, CardValidationError> {
        todo!("implement Card::with_guid")
    }

    /// Derive copy with new HTML content and recalculated hash.
    pub fn update_content(&self, new_apf_html: String) -> Result<Self, CardValidationError> {
        todo!("implement update_content")
    }

    /// Derive copy with new tags.
    pub fn with_tags(&self, tags: Vec<String>) -> Self {
        todo!("implement with_tags")
    }
}

impl SyncAction {
    /// Validate and construct. UPDATE/DELETE require anki_guid.
    pub fn new(
        action_type: SyncActionType,
        card: Card,
        anki_guid: Option<String>,
        reason: Option<String>,
    ) -> Result<Self, CardValidationError> {
        todo!("implement SyncAction validation")
    }

    /// True for UPDATE or DELETE.
    pub fn is_destructive(&self) -> bool {
        todo!("implement is_destructive")
    }

    /// True for DELETE.
    pub fn requires_confirmation(&self) -> bool {
        todo!("implement requires_confirmation")
    }

    /// Human-readable description: "UPDATE: my-slug (reason)".
    pub fn describe(&self) -> String {
        todo!("implement describe")
    }
}

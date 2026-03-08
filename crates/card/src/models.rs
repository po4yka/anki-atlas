use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Valid Anki note types.
pub const VALID_NOTE_TYPES: &[&str] = &["APF::Simple", "APF::Cloze", "Basic", "Cloze"];

/// Valid 2-letter language codes (must match common::Language variants).
const VALID_LANGUAGES: &[&str] = &["en", "ru", "de", "fr", "es", "it", "pt", "zh", "ja", "ko"];

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
#[derive(Debug, thiserror::Error)]
#[error("card validation failed: {messages:?}")]
pub struct CardValidationError {
    pub messages: Vec<String>,
}

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

fn is_valid_lang(lang: &str) -> bool {
    lang.len() == 2
        && lang.chars().all(|c| c.is_ascii_alphabetic())
        && VALID_LANGUAGES.contains(&lang)
}

fn is_valid_hash6(h: &str) -> bool {
    h.len() == 6 && h.chars().all(|c| c.is_ascii_hexdigit())
}

/// Push an error if `value` is empty.
fn require_non_empty(errors: &mut Vec<String>, field: &str, value: &str) {
    if value.is_empty() {
        errors.push(format!("{field} must not be empty"));
    }
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
        let mut errors = Vec::new();

        require_non_empty(&mut errors, "slug", &slug);
        require_non_empty(&mut errors, "slug_base", &slug_base);
        if !is_valid_lang(&lang) {
            errors.push(format!("lang must be a valid 2-letter language code, got '{lang}'"));
        }
        require_non_empty(&mut errors, "source_path", &source_path);
        require_non_empty(&mut errors, "source_anchor", &source_anchor);
        require_non_empty(&mut errors, "note_id", &note_id);
        require_non_empty(&mut errors, "note_title", &note_title);
        if let Some(ref h) = hash6 {
            if !is_valid_hash6(h) {
                errors.push(format!("hash6 must be exactly 6 hex chars, got '{h}'"));
            }
        }
        if let Some(d) = difficulty {
            if !(0.0..=1.0).contains(&d) {
                errors.push(format!("difficulty must be in [0.0, 1.0], got {d}"));
            }
        }

        if !errors.is_empty() {
            return Err(CardValidationError { messages: errors });
        }

        Ok(Self {
            slug,
            slug_base,
            lang,
            source_path,
            source_anchor,
            note_id,
            note_title,
            card_index,
            guid,
            hash6,
            obsidian_uri,
            difficulty,
            cognitive_load,
        })
    }

    /// Obsidian wikilink: [[folder/note#anchor]].
    pub fn anchor_url(&self) -> String {
        format!("[[{}#{}]]", self.source_path, self.source_anchor)
    }

    /// True if note_id and source_path are non-empty.
    pub fn is_linked_to_note(&self) -> bool {
        !self.note_id.is_empty() && !self.source_path.is_empty()
    }

    /// Derive a copy with the given Anki GUID.
    pub fn with_guid(&self, guid: String) -> Self {
        let mut copy = self.clone();
        copy.guid = Some(guid);
        copy
    }

    /// Derive a copy with the given content hash.
    pub fn with_hash(&self, hash6: String) -> Self {
        let mut copy = self.clone();
        copy.hash6 = Some(hash6);
        copy
    }

    /// Derive a copy with the given Obsidian URI.
    pub fn with_obsidian_uri(&self, uri: String) -> Self {
        let mut copy = self.clone();
        copy.obsidian_uri = Some(uri);
        copy
    }

    /// Derive a copy with FSRS metadata.
    pub fn with_fsrs_metadata(&self, difficulty: f64, cognitive_load: CognitiveLoad) -> Self {
        let mut copy = self.clone();
        copy.difficulty = Some(difficulty);
        copy.cognitive_load = Some(cognitive_load);
        copy
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
        let mut errors = Vec::new();

        if slug.is_empty() {
            errors.push("slug must not be empty".into());
        } else if slug.len() < 3 {
            errors.push(format!("slug must be at least 3 chars, got '{slug}'"));
        }
        if !is_valid_lang(&language) {
            errors.push(format!(
                "language must be a valid 2-letter language code, got '{language}'"
            ));
        }
        if apf_html.trim().len() < 10 {
            errors.push("apf_html must be at least 10 chars after trim".into());
        }
        if !VALID_NOTE_TYPES.contains(&note_type.as_str()) {
            errors.push(format!("note_type must be one of {VALID_NOTE_TYPES:?}, got '{note_type}'"));
        }
        if manifest.slug != slug {
            errors.push(format!(
                "slug mismatch: card slug '{}' != manifest slug '{}'",
                slug, manifest.slug
            ));
        }
        if manifest.lang != language {
            errors.push(format!(
                "language mismatch: card language '{}' != manifest lang '{}'",
                language, manifest.lang
            ));
        }
        for (i, tag) in tags.iter().enumerate() {
            if tag.is_empty() {
                errors.push(format!("tag at index {i} must not be empty"));
            }
        }

        if !errors.is_empty() {
            return Err(CardValidationError { messages: errors });
        }

        Ok(Self {
            slug,
            language,
            apf_html,
            manifest,
            note_type,
            tags,
            anki_guid,
        })
    }

    /// True if anki_guid is None.
    pub fn is_new(&self) -> bool {
        self.anki_guid.is_none()
    }

    /// SHA-256[:6] of "{apf_html}|{note_type}|{sorted,tags}".
    pub fn content_hash(&self) -> String {
        let mut sorted_tags = self.tags.clone();
        sorted_tags.sort();
        let input = format!("{}|{}|{}", self.apf_html, self.note_type, sorted_tags.join(","));
        let hash = Sha256::digest(input.as_bytes());
        hash[..3].iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Derive copy with Anki GUID set on both Card and Manifest.
    pub fn with_guid(&self, guid: String) -> Result<Self, CardValidationError> {
        let mut copy = self.clone();
        copy.anki_guid = Some(guid.clone());
        copy.manifest = copy.manifest.with_guid(guid);
        Ok(copy)
    }

    /// Derive copy with new HTML content and recalculated hash.
    pub fn update_content(&self, new_apf_html: String) -> Result<Self, CardValidationError> {
        if new_apf_html.trim().len() < 10 {
            return Err(CardValidationError {
                messages: vec!["apf_html must be at least 10 chars after trim".into()],
            });
        }
        let mut copy = self.clone();
        copy.apf_html = new_apf_html;
        Ok(copy)
    }

    /// Derive copy with new tags.
    pub fn with_tags(&self, tags: Vec<String>) -> Self {
        let mut copy = self.clone();
        copy.tags = tags;
        copy
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
        let needs_guid = matches!(action_type, SyncActionType::Update | SyncActionType::Delete);
        if needs_guid && anki_guid.is_none() {
            return Err(CardValidationError {
                messages: vec!["anki_guid is required for UPDATE/DELETE actions".into()],
            });
        }

        Ok(Self {
            action_type,
            card,
            anki_guid,
            reason,
        })
    }

    /// True for UPDATE or DELETE.
    pub fn is_destructive(&self) -> bool {
        matches!(self.action_type, SyncActionType::Update | SyncActionType::Delete)
    }

    /// True for DELETE.
    pub fn requires_confirmation(&self) -> bool {
        matches!(self.action_type, SyncActionType::Delete)
    }

    /// Human-readable description: "UPDATE: my-slug (reason)".
    pub fn describe(&self) -> String {
        let action = match self.action_type {
            SyncActionType::Create => "CREATE",
            SyncActionType::Update => "UPDATE",
            SyncActionType::Delete => "DELETE",
            SyncActionType::Skip => "SKIP",
        };
        match &self.reason {
            Some(reason) => format!("{action}: {} ({reason})", self.card.slug),
            None => format!("{action}: {}", self.card.slug),
        }
    }
}

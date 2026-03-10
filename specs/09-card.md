# Spec: crate `card`

## Source Reference

Current Rust implementation: `crates/card/`
Historical rewrite input: `packages/card/` (models.py, slug.py, mapping.py, registry.py)

## Purpose

Core card domain: value objects (`Card`, `CardManifest`, `SyncAction`), deterministic slug generation with validation, card-to-note mapping types, and a SQLite-backed card registry with schema migrations. All domain types are immutable (no interior mutability). The registry provides CRUD for cards and notes used during obsidian-to-Anki sync.

## Dependencies

```toml
[dependencies]
common = { path = "../common" }
serde = { version = "1", features = ["derive"] }
sha2 = "0.10"
regex = "1"
unicode-normalization = "0.1"
thiserror = "2"
tracing = "0.1"
chrono = { version = "0.4", features = ["serde"] }
rusqlite = { version = "0.32", features = ["bundled"] }

[dev-dependencies]
tempfile = "3"
```

## Public API

### Models (`src/models.rs`)

```rust
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

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

/// Value object linking a card to its source Obsidian note.
/// Immutable -- use `with_*` methods to derive modified copies.
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

/// Validation error for card domain types.
#[derive(Debug, thiserror::Error)]
#[error("card validation failed: {messages:?}")]
pub struct CardValidationError {
    pub messages: Vec<String>,
}

impl CardManifest {
    /// Validate and construct. Returns Err with all validation failures.
    pub fn new(/* all fields */) -> Result<Self, CardValidationError>;

    /// Obsidian wikilink: [[folder/note#anchor]].
    pub fn anchor_url(&self) -> String;

    /// True if note_id and source_path are non-empty.
    pub fn is_linked_to_note(&self) -> bool;

    /// Derive a copy with the given Anki GUID.
    pub fn with_guid(&self, guid: String) -> Self;

    /// Derive a copy with the given content hash.
    pub fn with_hash(&self, hash6: String) -> Self;

    /// Derive a copy with the given Obsidian URI.
    pub fn with_obsidian_uri(&self, uri: String) -> Self;

    /// Derive a copy with FSRS metadata.
    pub fn with_fsrs_metadata(&self, difficulty: f64, cognitive_load: CognitiveLoad) -> Self;
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

impl Card {
    /// Validate and construct.
    pub fn new(/* all fields */) -> Result<Self, CardValidationError>;

    /// True if anki_guid is None.
    pub fn is_new(&self) -> bool;

    /// SHA-256[:6] of "{apf_html}|{note_type}|{sorted,tags}".
    pub fn content_hash(&self) -> String;

    /// Derive copy with Anki GUID set on both Card and Manifest.
    pub fn with_guid(&self, guid: String) -> Result<Self, CardValidationError>;

    /// Derive copy with new HTML content and recalculated hash.
    pub fn update_content(&self, new_apf_html: String) -> Result<Self, CardValidationError>;

    /// Derive copy with new tags.
    pub fn with_tags(&self, tags: Vec<String>) -> Self;
}

/// A sync action for a card.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyncAction {
    pub action_type: SyncActionType,
    pub card: Card,
    pub anki_guid: Option<String>,
    pub reason: Option<String>,
}

impl SyncAction {
    /// Validate and construct. UPDATE/DELETE require anki_guid.
    pub fn new(
        action_type: SyncActionType,
        card: Card,
        anki_guid: Option<String>,
        reason: Option<String>,
    ) -> Result<Self, CardValidationError>;

    /// True for UPDATE or DELETE.
    pub fn is_destructive(&self) -> bool;

    /// True for DELETE.
    pub fn requires_confirmation(&self) -> bool;

    /// Human-readable description: "UPDATE: my-slug (reason)".
    pub fn describe(&self) -> String;
}
```

### Slug Service (`src/slug.rs`)

```rust
/// Maximum length for a single slug component (topic or keyword).
pub const MAX_COMPONENT_LENGTH: usize = 50;
/// Maximum total slug length.
pub const MAX_SLUG_LENGTH: usize = 100;

/// Stateless slug generation utilities.
pub struct SlugService;

impl SlugService {
    /// Convert arbitrary text to a URL-friendly slug.
    /// NFKD normalize, strip combining marks, lowercase, replace separators
    /// with hyphens, remove non-[a-z0-9-], collapse multiple hyphens,
    /// truncate to MAX_COMPONENT_LENGTH at word boundary.
    pub fn slugify(text: &str) -> String;

    /// SHA-256[:length] of content. length must be in 1..=64.
    pub fn compute_hash(content: &str, length: usize) -> Result<String, SlugError>;

    /// Generate slug: "{topic}-{keyword}-{index}-{lang}".
    /// Truncates components to fit MAX_SLUG_LENGTH.
    pub fn generate_slug(topic: &str, keyword: &str, index: u32, lang: &str) -> Result<String, SlugError>;

    /// Generate slug base without language suffix: "{topic}-{keyword}-{index}".
    pub fn generate_slug_base(topic: &str, keyword: &str, index: u32) -> Result<String, SlugError>;

    /// Deterministic GUID: SHA-256[:32] of "{slug}:{source_path}".
    pub fn generate_deterministic_guid(slug: &str, source_path: &str) -> Result<String, SlugError>;

    /// Extract components from a slug -> { topic, keyword, index, lang }.
    pub fn extract_components(slug: &str) -> SlugComponents;

    /// Validate: >= 3 chars, matches [a-z0-9][a-z0-9-]*[a-z0-9], no "--",
    /// ends with "-{2-letter-lang}".
    pub fn is_valid_slug(slug: &str) -> bool;

    /// SHA-256[:12] of "{front.trim()}|{back.trim()}".
    pub fn compute_content_hash(front: &str, back: &str) -> String;

    /// SHA-256[:6] of "{note_type}|{sorted,tags}".
    pub fn compute_metadata_hash(note_type: &str, tags: &[String]) -> String;
}

#[derive(Debug, Clone, Default)]
pub struct SlugComponents {
    pub topic: String,
    pub keyword: String,
    pub index: u32,
    pub lang: String,
}

#[derive(Debug, thiserror::Error)]
pub enum SlugError {
    #[error("invalid hash length: {0} (must be 1..=64)")]
    InvalidHashLength(usize),
    #[error("invalid index: must be non-negative")]
    InvalidIndex,
    #[error("invalid lang: must be exactly 2 characters, got '{0}'")]
    InvalidLang(String),
    #[error("empty input: {field}")]
    EmptyInput { field: &'static str },
}
```

### Mapping (`src/mapping.rs`)

```rust
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
    pub fn is_synced(&self) -> bool;

    /// Construct from a CardEntry.
    pub fn from_card_entry(entry: &super::registry::CardEntry) -> Self;
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
    pub fn card_count(&self) -> usize;
    pub fn synced_count(&self) -> usize;
    pub fn unsynced_count(&self) -> usize;
}
```

### Card Registry (`src/registry.rs`)

```rust
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
    pub fn open(db_path: &str) -> Result<Self, RegistryError>;

    /// Close the connection.
    pub fn close(self);

    // --- Card CRUD ---

    /// Insert a card. Returns Ok(true) on success, Ok(false) if slug exists.
    pub fn add_card(&self, entry: &CardEntry) -> Result<bool, RegistryError>;

    /// Get card by slug.
    pub fn get_card(&self, slug: &str) -> Result<Option<CardEntry>, RegistryError>;

    /// Update card by slug. Returns Ok(true) if updated.
    pub fn update_card(&self, entry: &CardEntry) -> Result<bool, RegistryError>;

    /// Delete card by slug. Returns Ok(true) if deleted.
    pub fn delete_card(&self, slug: &str) -> Result<bool, RegistryError>;

    /// Find cards by optional filters (note_id, source_path, content_hash).
    pub fn find_cards(
        &self,
        note_id: Option<&str>,
        source_path: Option<&str>,
        content_hash: Option<&str>,
    ) -> Result<Vec<CardEntry>, RegistryError>;

    // --- Note CRUD ---

    pub fn add_note(&self, entry: &NoteEntry) -> Result<bool, RegistryError>;
    pub fn get_note(&self, note_id: &str) -> Result<Option<NoteEntry>, RegistryError>;
    pub fn list_notes(&self) -> Result<Vec<NoteEntry>, RegistryError>;

    // --- Stats ---

    pub fn card_count(&self) -> Result<usize, RegistryError>;
    pub fn note_count(&self) -> Result<usize, RegistryError>;

    // --- Mapping ---

    /// Get all cards for a given note.
    pub fn get_mapping(&self, note_id: &str) -> Result<Vec<CardEntry>, RegistryError>;

    /// Replace all cards for a note (delete + re-insert).
    pub fn update_mapping(&self, note_id: &str, cards: &[CardEntry]) -> Result<(), RegistryError>;
}
```

## Internal Details

### CardManifest validation (in `new()`)
- `slug`, `slug_base`, `source_path`, `source_anchor`, `note_id`, `note_title` must be non-empty.
- `lang` must be exactly 2 ASCII letters and in the valid languages set from `common`.
- `hash6`, if present, must be exactly 6 hex chars.
- `difficulty`, if present, must be in `[0.0, 1.0]`.
- Collect all errors and return a single `CardValidationError` with all messages.

### Card validation (in `new()`)
- `slug` must be non-empty and >= 3 chars.
- `language` must be 2 chars and valid.
- `apf_html` must be non-empty and >= 10 chars after trim.
- `note_type` must be in `VALID_NOTE_TYPES`.
- `manifest.lang == language` and `manifest.slug == slug`.
- Tags must be non-empty strings.

### Slug truncation algorithm
When `generate_slug` produces a slug exceeding `MAX_SLUG_LENGTH`:
1. Compute available space: `MAX_SLUG_LENGTH - len("-{index}-{lang}")`.
2. Split available in half.
3. Truncate topic and keyword to half, stripping trailing hyphens.

### Slug component extraction
Parse from the right:
1. Last segment (2 alpha chars) = lang.
2. Next segment (all digits) = index.
3. Remaining string split in half = topic + keyword.

### Registry schema migrations
- On open, check for `schema_version` table.
  - If missing: create full schema (cards, notes, schema_version, indexes).
  - If present: read version, apply incremental migrations (v1->v2 adds notes table).
- SQLite pragmas: `journal_mode=WAL`, `foreign_keys=ON`.

### Tags serialization in SQLite
- Stored as comma-separated string in the `tags` column.
- Empty tags list = empty string.
- Deserialized by splitting on `,`.

### Datetime handling
- Stored as ISO 8601 strings in SQLite.
- `created_at` defaults to current UTC time if not provided.
- `updated_at` always set to current UTC time on update.

## Acceptance Criteria

- [ ] `cargo test -p card` passes
- [ ] `cargo clippy -p card -- -D warnings` clean
- [ ] All public types are `Send + Sync`
- [ ] `CardManifest::new` rejects empty slug, invalid lang, invalid hash6
- [ ] `CardManifest::new` collects all errors (not just the first)
- [ ] `Card::new` rejects mismatched manifest.lang vs card.language
- [ ] `Card::content_hash` is deterministic and changes when content changes
- [ ] `Card::with_guid` updates both `anki_guid` and `manifest.guid`
- [ ] `SyncAction::new` requires anki_guid for UPDATE and DELETE
- [ ] `SlugService::slugify` handles Unicode (NFKD normalization, accent stripping)
- [ ] `SlugService::slugify` truncates at word boundary within MAX_COMPONENT_LENGTH
- [ ] `SlugService::generate_slug` stays within MAX_SLUG_LENGTH
- [ ] `SlugService::is_valid_slug` rejects: empty, < 3 chars, double hyphens, no lang suffix
- [ ] `SlugService::extract_components` roundtrips with `generate_slug`
- [ ] `CardRegistry::open(":memory:")` creates schema automatically
- [ ] `CardRegistry` add_card returns false on duplicate slug
- [ ] `CardRegistry` update_card returns false for non-existent slug
- [ ] `CardRegistry` schema migration from v1 to v2 adds notes table
- [ ] `CardMappingEntry::is_synced` returns true iff anki_note_id is Some
- [ ] `NoteMapping::unsynced_count` = card_count - synced_count

# Spec: crate `anki-reader`

## Source Reference
Python: `packages/anki/` (models.py, reader.py, normalizer.py, connect.py)

## Purpose
Read and normalize Anki collections from SQLite databases and communicate with a running Anki instance via the AnkiConnect HTTP API. Provides strongly-typed models for decks, notes, cards, review logs, and card statistics. Includes HTML-to-text normalization for search indexing. The SQLite reader copies the collection file to a temp location to avoid locking conflicts with a running Anki.

## Dependencies
```toml
[dependencies]
common = { path = "../common" }
rusqlite = { version = "0.32", features = ["bundled"] }
reqwest = { version = "0.12", features = ["json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
chrono = { version = "0.4", features = ["serde"] }
tempfile = "3"
regex = "1"
tokio = { version = "1", features = ["fs"] }
tracing = "0.1"
once_cell = "1"

[dev-dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
wiremock = "0.6"
```

## Public API

### Models (`src/models.rs`)

```rust
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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
    /// Raw field values split by \x1f.
    pub fields: Vec<String>,
    /// Named field -> value mapping.
    pub fields_json: std::collections::HashMap<String, String>,
    /// Original unsplit field string.
    pub raw_fields: Option<String>,
    /// Normalized plain-text for search indexing (populated by normalizer).
    pub normalized_text: String,
    /// Modification timestamp (seconds since epoch).
    pub mtime: i64,
    /// Update sequence number.
    pub usn: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnkiCard {
    pub card_id: i64,
    pub note_id: i64,
    pub deck_id: i64,
    /// Card ordinal within note (0-based).
    pub ord: i32,
    pub due: Option<i32>,
    /// Interval in days.
    pub ivl: i32,
    /// Ease factor in permille (e.g. 2500 = 250%).
    pub ease: i32,
    pub lapses: i32,
    pub reps: i32,
    /// Queue: -3=user buried, -2=sched buried, -1=suspended, 0=new, 1=learning, 2=review, 3=in learning.
    pub queue: i32,
    /// Type: 0=new, 1=learning, 2=review, 3=relearning.
    pub card_type: i32,
    pub mtime: i64,
    pub usn: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnkiRevlogEntry {
    /// Timestamp in milliseconds.
    pub id: i64,
    pub card_id: i64,
    pub usn: i32,
    /// 1=again, 2=hard, 3=good, 4=easy.
    pub button_chosen: i32,
    /// New interval.
    pub interval: i64,
    /// Previous interval.
    pub last_interval: i64,
    /// New ease factor.
    pub ease: i32,
    /// Time spent on review in milliseconds.
    pub time_ms: i64,
    /// 0=learn, 1=review, 2=relearn, 3=filtered.
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
    fn default() -> Self; // schema_version defaults to 11
}
```

### Reader (`src/reader.rs`)

```rust
use std::path::Path;
use common::error::Result;
use crate::models::*;

/// Read an Anki collection from a SQLite database file.
///
/// Copies the file to a temp location before opening to avoid
/// lock contention with a running Anki instance.
pub struct AnkiReader {
    collection_path: std::path::PathBuf,
    // internal: temp file handle, rusqlite::Connection
}

impl AnkiReader {
    /// Create a new reader. Validates that the collection file exists.
    pub fn new(collection_path: impl AsRef<Path>) -> Result<Self>;

    /// Open the database (copy to temp, connect). Must be called before read methods.
    pub fn open(&mut self) -> Result<()>;

    /// Close the database and clean up temp file.
    pub fn close(&mut self);

    /// Read the complete collection (decks + models + notes + cards + stats).
    pub fn read_collection(&self) -> Result<AnkiCollection>;

    /// Read all decks. Supports both legacy (col.decks JSON) and modern (decks table) schemas.
    pub fn read_decks(&self) -> Result<Vec<AnkiDeck>>;

    /// Read all note types/models. Supports legacy (col.models JSON) and modern (notetypes table).
    pub fn read_models(&self) -> Result<Vec<AnkiModel>>;

    /// Read all notes. Uses models for field name mapping.
    pub fn read_notes(&self, models: &[AnkiModel]) -> Result<Vec<AnkiNote>>;

    /// Read all cards.
    pub fn read_cards(&self) -> Result<Vec<AnkiCard>>;

    /// Read review log entries.
    pub fn read_revlog(&self) -> Result<Vec<AnkiRevlogEntry>>;

    /// Compute aggregated card statistics from the revlog.
    pub fn compute_card_stats(&self) -> Result<Vec<CardStats>>;
}

/// Convenience function: open, read, close.
pub fn read_anki_collection(path: impl AsRef<Path>) -> Result<AnkiCollection>;
```

RAII pattern: implement `Drop` for `AnkiReader` to close connection and delete temp file.

### Normalizer (`src/normalizer.rs`)

```rust
use crate::models::{AnkiCard, AnkiDeck, AnkiNote};

/// Strip HTML tags from text.
/// If `preserve_code` is true, content of `<code>` and `<pre>` blocks is
/// preserved (wrapped in backticks in the output).
/// Handles: cloze deletions (extracts answer), `<br>` -> newline,
/// `&nbsp;` -> space, HTML entity decoding.
pub fn strip_html(text: &str, preserve_code: bool) -> String;

/// Normalize whitespace: collapse runs of spaces/tabs to single space,
/// preserve single newlines, trim each line, remove blank lines.
pub fn normalize_whitespace(text: &str) -> String;

/// Classify a field name as "front", "back", "extra", or "other".
///
/// Front: front, question, expression, word, term, prompt
/// Back: back, answer, meaning, definition, response, reading
/// Extra: extra, notes, hint, example, examples, context
pub fn classify_field(name: &str) -> &'static str;

/// Normalize a single note to searchable text.
///
/// Output template:
///     Front: ...
///     Back: ...
///     Extra: ...
///     Tags: tag1, tag2
///     Decks: Deck1, Deck2
pub fn normalize_note(note: &AnkiNote, deck_names: Option<&[String]>) -> String;

/// Normalize all notes in-place, populating `normalized_text`.
pub fn normalize_notes(
    notes: &mut [AnkiNote],
    deck_map: &std::collections::HashMap<i64, String>,
    card_deck_map: &std::collections::HashMap<i64, Vec<i64>>,
);

/// Build deck_id -> deck_name mapping.
pub fn build_deck_map(decks: &[AnkiDeck]) -> std::collections::HashMap<i64, String>;

/// Build note_id -> vec of deck_ids mapping from cards.
pub fn build_card_deck_map(cards: &[AnkiCard]) -> std::collections::HashMap<i64, Vec<i64>>;
```

### AnkiConnect Client (`src/connect.rs`)

```rust
use common::error::Result;
use serde_json::Value;
use std::collections::HashMap;

/// Default AnkiConnect URL.
pub const ANKI_CONNECT_URL: &str = "http://localhost:8765";
/// AnkiConnect protocol version.
pub const ANKI_CONNECT_VERSION: u32 = 6;
/// Default request timeout in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Async HTTP client for the AnkiConnect API.
pub struct AnkiConnectClient {
    url: String,
    timeout: std::time::Duration,
    client: reqwest::Client,
}

impl AnkiConnectClient {
    pub fn new(url: &str, timeout_secs: u64) -> Self;

    /// Create with default URL and timeout.
    pub fn default() -> Self;

    /// Send a raw action request to AnkiConnect.
    /// Returns the `result` field or an `AnkiConnect` error.
    pub async fn invoke(&self, action: &str, params: Option<Value>) -> Result<Value>;

    // -- Connection --
    pub async fn ping(&self) -> bool;
    pub async fn version(&self) -> Option<u32>;

    // -- Decks --
    pub async fn deck_names(&self) -> Result<Vec<String>>;
    pub async fn create_deck(&self, name: &str) -> Result<i64>;
    pub async fn delete_decks(&self, names: &[String], cards_too: bool) -> Result<()>;

    // -- Notes --
    pub async fn find_notes(&self, query: &str) -> Result<Vec<i64>>;
    pub async fn notes_info(&self, note_ids: &[i64]) -> Result<Vec<Value>>;
    pub async fn add_note(
        &self,
        deck_name: &str,
        model_name: &str,
        fields: &HashMap<String, String>,
        tags: &[String],
        allow_duplicate: bool,
    ) -> Result<Option<i64>>;
    pub async fn update_note_fields(&self, note_id: i64, fields: &HashMap<String, String>) -> Result<()>;
    pub async fn delete_notes(&self, note_ids: &[i64]) -> Result<()>;

    // -- Tags --
    pub async fn get_tags(&self) -> Result<Vec<String>>;
    pub async fn add_tags(&self, note_ids: &[i64], tags: &str) -> Result<()>;
    pub async fn remove_tags(&self, note_ids: &[i64], tags: &str) -> Result<()>;

    // -- Models --
    pub async fn model_names(&self) -> Result<Vec<String>>;
    pub async fn model_field_names(&self, model_name: &str) -> Result<Vec<String>>;

    // -- Sync --
    pub async fn sync(&self) -> Result<()>;
}
```

### Module root (`src/lib.rs`)

```rust
pub mod connect;
pub mod models;
pub mod normalizer;
pub mod reader;

pub use connect::AnkiConnectClient;
pub use models::*;
pub use reader::{read_anki_collection, AnkiReader};
```

## Internal Details

### Schema detection
The reader checks for a `notetypes` table in `sqlite_master` to distinguish modern Anki schemas (>= 2.1.28) from legacy schemas. Legacy schemas store decks and models as JSON blobs in the `col` table; modern schemas use separate `decks`, `notetypes`, `fields`, and `templates` tables.

### Temp file copy
The reader copies the `.anki2` file to a `tempfile::NamedTempFile` before opening. This is critical because Anki holds a WAL lock on the database. The `Drop` implementation ensures cleanup even on panic.

### Regex compilation
The normalizer uses 5 regex patterns (cloze, HTML tags, whitespace, code blocks, inline code). These must be compiled once using `once_cell::sync::Lazy` or `std::sync::LazyLock` and reused across calls.

### Cloze deletion pattern
`{{c\d+::([^}]+?)(?:::[^}]+)?}}` -- extracts the answer text (first capture group), discarding the cloze number and optional hint.

### HTML stripping order
1. Save `<pre>` and `<code>` content as placeholders (if `preserve_code`).
2. Expand cloze deletions.
3. Replace `<br>`, `<p>`, `<div>`, `<li>` with newlines.
4. Strip all remaining HTML tags.
5. Decode HTML entities (`&amp;`, `&lt;`, etc.).
6. Restore code blocks (wrapped in backticks).
7. Collapse whitespace.

### Field classification fallback
When no fields match the front/back/extra name sets, the first field is treated as front, the second as back, and remaining as extra. This handles custom note types with non-standard field names.

### AnkiConnect error handling
The `invoke` method checks the `error` field in the JSON response. If non-null, it raises `AnkiAtlasError::AnkiConnect`. Connection failures (`reqwest::Error` with `is_connect()`) produce a specific "Is Anki running?" message. The `add_note` method returns `None` (instead of error) when the error message contains "duplicate".

### Note field parsing
Fields in Anki notes are stored as a single string separated by `\x1f` (unit separator, ASCII 31). The reader splits on this character and maps positional values to named fields using the model's field definitions.

## Acceptance Criteria
- [ ] `cargo test -p anki-reader` passes
- [ ] `cargo clippy -p anki-reader -- -D warnings` clean
- [ ] All public types are `Send + Sync`
- [ ] `AnkiReader` reads a test `.anki2` file with legacy schema (col.decks JSON)
- [ ] `AnkiReader` reads a test `.anki2` file with modern schema (decks table)
- [ ] Temp file is created before open and deleted after close / drop
- [ ] `read_notes` correctly splits fields on `\x1f` and maps to named fields
- [ ] `read_notes` parses space-separated tags
- [ ] `compute_card_stats` aggregates review counts, avg ease, fail rate, last review timestamp
- [ ] `strip_html("<b>hello</b>", false)` returns `"hello"`
- [ ] `strip_html` preserves code block content when `preserve_code = true`
- [ ] `strip_html` expands cloze deletions: `"{{c1::answer::hint}}"` -> `"answer"`
- [ ] `classify_field("Front")` returns `"front"`, `classify_field("xyz")` returns `"other"`
- [ ] `normalize_note` produces deterministic `Front: ...\nBack: ...\nTags: ...` output
- [ ] `build_deck_map` and `build_card_deck_map` produce correct mappings
- [ ] `AnkiConnectClient::invoke` sends correct JSON payload with version 6
- [ ] `AnkiConnectClient::ping` returns false when connection is refused (use wiremock)
- [ ] `AnkiConnectClient::add_note` returns `None` on duplicate error
- [ ] Empty inputs (no decks, no notes, empty tags) are handled gracefully without panics

# Spec: crate `common`

## Source Reference
Python: `packages/common/` (config.py, types.py, exceptions.py, logging.py)

## Purpose
Foundation crate providing configuration loading from environment variables, shared domain types (newtypes, enums), a structured error hierarchy, and correlation-ID-aware structured logging. This crate has zero database or network dependencies -- it is pure logic and types.

## Dependencies
```toml
[dependencies]
config = "0.15"               # env/file config loading
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
uuid = { version = "1", features = ["v4"] }
strum = { version = "0.27", features = ["derive"] }
once_cell = "1"

[dev-dependencies]
temp-env = "0.3"              # set env vars in tests
```

## Public API

### Types (`src/types.rs`)

```rust
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

/// Supported card languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, EnumString, Display)]
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
```

### Config (`src/config.rs`)

```rust
use serde::Deserialize;

/// Qdrant quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    None,
    Scalar,
    Binary,
}

/// Application settings loaded from env vars prefixed `ANKIATLAS_`.
/// Example: `ANKIATLAS_POSTGRES_URL` maps to `postgres_url`.
#[derive(Debug, Clone, Deserialize)]
pub struct Settings {
    // -- Database --
    #[serde(default = "defaults::postgres_url")]
    pub postgres_url: String,

    // -- Vector store --
    #[serde(default = "defaults::qdrant_url")]
    pub qdrant_url: String,
    #[serde(default = "defaults::qdrant_quantization")]
    pub qdrant_quantization: Quantization,
    #[serde(default)]
    pub qdrant_on_disk: bool,

    // -- Async jobs --
    #[serde(default = "defaults::redis_url")]
    pub redis_url: String,
    #[serde(default = "defaults::job_queue_name")]
    pub job_queue_name: String,
    #[serde(default = "defaults::job_result_ttl_seconds")]
    pub job_result_ttl_seconds: u32,
    #[serde(default = "defaults::job_max_retries")]
    pub job_max_retries: u32,

    // -- Embeddings --
    #[serde(default = "defaults::embedding_provider")]
    pub embedding_provider: String,
    #[serde(default = "defaults::embedding_model")]
    pub embedding_model: String,
    #[serde(default = "defaults::embedding_dimension")]
    pub embedding_dimension: u32,
    #[serde(default)]
    pub rerank_enabled: bool,
    #[serde(default = "defaults::rerank_model")]
    pub rerank_model: String,
    #[serde(default = "defaults::rerank_top_n")]
    pub rerank_top_n: u32,
    #[serde(default = "defaults::rerank_batch_size")]
    pub rerank_batch_size: u32,

    // -- API --
    #[serde(default = "defaults::api_host")]
    pub api_host: String,
    #[serde(default = "defaults::api_port")]
    pub api_port: u16,
    pub api_key: Option<String>,
    #[serde(default)]
    pub debug: bool,

    // -- Anki source --
    pub anki_collection_path: Option<String>,
}

impl Settings {
    /// Load settings from environment variables and optional `.env` file.
    /// Validates all fields after loading.
    pub fn load() -> Result<Self, ConfigError>;

    /// Validate all fields. Called automatically by `load()`.
    pub fn validate(&self) -> Result<(), ConfigError>;
}

/// Return a lazily-initialized, globally cached `&'static Settings`.
/// Panics if `Settings::load()` has never been called successfully.
/// For non-panic usage, call `Settings::load()` directly.
pub fn get_settings() -> &'static Settings;
```

Validation rules (enforced in `validate()`):
- `embedding_dimension` must be in `{384, 768, 1024, 1536, 3072}` unless `embedding_provider == "mock"`
- `postgres_url` must start with `postgresql://` or `postgres://`
- `qdrant_url` must start with `http://` or `https://`
- `redis_url` must start with `redis://` or `rediss://`
- `job_result_ttl_seconds`, `job_max_retries`, `rerank_top_n`, `rerank_batch_size` must be > 0
- `embedding_dimension` must be > 0

### Errors (`src/error.rs`)

```rust
use std::collections::HashMap;
use thiserror::Error;

/// Optional context attached to any error variant.
pub type ErrorContext = HashMap<String, String>;

#[derive(Debug, Error)]
pub enum AnkiAtlasError {
    // -- Database --
    #[error("database connection failed: {message}")]
    DatabaseConnection { message: String, context: ErrorContext },

    #[error("migration failed: {message}")]
    Migration { message: String, context: ErrorContext },

    // -- Vector store --
    #[error("vector store connection failed: {message}")]
    VectorStoreConnection { message: String, context: ErrorContext },

    #[error("collection operation failed: {message}")]
    Collection { message: String, context: ErrorContext },

    #[error("dimension mismatch on '{collection}': expected {expected}, got {actual}")]
    DimensionMismatch {
        collection: String,
        expected: u32,
        actual: u32,
    },

    // -- Embedding --
    #[error("embedding error: {message}")]
    Embedding { message: String, context: ErrorContext },

    #[error("embedding API error: {message}")]
    EmbeddingApi { message: String, context: ErrorContext },

    #[error("embedding timeout: {message}")]
    EmbeddingTimeout { message: String, context: ErrorContext },

    #[error("embedding model changed: '{stored}' -> '{current}'. Use --force-reindex.")]
    EmbeddingModelChanged { stored: String, current: String },

    // -- Sync --
    #[error("sync error: {message}")]
    Sync { message: String, context: ErrorContext },

    #[error("collection not found: {message}")]
    CollectionNotFound { message: String, context: ErrorContext },

    #[error("sync conflict: {message}")]
    SyncConflict { message: String, context: ErrorContext },

    // -- Anki --
    #[error("AnkiConnect error: {message}")]
    AnkiConnect { message: String, context: ErrorContext },

    #[error("Anki reader error: {message}")]
    AnkiReader { message: String, context: ErrorContext },

    // -- Config --
    #[error("configuration error: {message}")]
    Configuration { message: String, context: ErrorContext },

    // -- CRUD --
    #[error("not found: {message}")]
    NotFound { message: String, context: ErrorContext },

    #[error("conflict: {message}")]
    Conflict { message: String, context: ErrorContext },

    // -- Card generation --
    #[error("card generation error: {message}")]
    CardGeneration { message: String, context: ErrorContext },

    #[error("card validation error: {message}")]
    CardValidation { message: String, context: ErrorContext },

    // -- Provider --
    #[error("provider error: {message}")]
    Provider { message: String, context: ErrorContext },

    // -- Obsidian --
    #[error("obsidian parse error: {message}")]
    ObsidianParse { message: String, context: ErrorContext },

    // -- Jobs --
    #[error("job backend unavailable: {message}")]
    JobBackendUnavailable { message: String, context: ErrorContext },
}

/// Alias used throughout the codebase.
pub type Result<T> = std::result::Result<T, AnkiAtlasError>;
```

Also expose a convenience trait for adding context:

```rust
pub trait WithContext {
    fn with_context(self, key: impl Into<String>, value: impl Into<String>) -> Self;
}
```

Implement `WithContext` for `AnkiAtlasError` on variants that carry `ErrorContext`.

### Logging (`src/logging.rs`)

```rust
use std::io;

/// Initialize the global tracing subscriber.
/// Call once per process entry point.
///
/// - `debug`: if true, set level to DEBUG; otherwise INFO.
/// - `json_output`: if true, emit JSON lines; otherwise human-readable.
/// - `writer`: output destination (defaults to stderr).
pub fn configure_logging(debug: bool, json_output: bool, writer: impl io::Write + Send + 'static);

/// Correlation ID stored in a task-local (tokio) or thread-local.
/// Returns `None` if not set.
pub fn get_correlation_id() -> Option<String>;

/// Set or generate a correlation ID. Returns the ID that was set.
pub fn set_correlation_id(id: Option<String>) -> String;

/// Clear the correlation ID.
pub fn clear_correlation_id();
```

Implementation notes:
- Use `tracing_subscriber::fmt` with a `Layer` that injects the correlation ID into every span/event via a custom `Layer` impl or `tracing_subscriber::filter`.
- Store the correlation ID in a `tokio::task_local!` macro (or `thread_local!` for sync code).
- The JSON renderer uses `tracing_subscriber::fmt::format::Json`.
- The console renderer uses `tracing_subscriber::fmt::format::Full`.

### Module root (`src/lib.rs`)

```rust
pub mod config;
pub mod error;
pub mod logging;
pub mod types;

// Re-export key items at crate root for ergonomics.
pub use config::{get_settings, Settings};
pub use error::{AnkiAtlasError, Result};
pub use types::{CardId, DeckName, Language, NoteId, SlugStr};
```

## Internal Details

### Config loading strategy
Use the `config` crate with this source chain (last wins):
1. Compiled-in defaults (via `defaults` module of default functions)
2. `.env` file in CWD (optional, not an error if missing)
3. Environment variables with `ANKIATLAS_` prefix (case-insensitive, underscore-separated)

The `get_settings()` function uses `once_cell::sync::Lazy` to cache the first successful load. Tests should use `Settings::load()` directly with `temp_env` to override env vars.

### Dimension validation edge case
When `embedding_provider` is `"mock"`, any positive dimension is accepted. For all other providers, only `{384, 768, 1024, 1536, 3072}` are valid.

### Error context pattern
The `ErrorContext` (`HashMap<String, String>`) is optional metadata for debugging. It is not part of the Display output but is available via the struct field. Structured logging should emit it.

## Acceptance Criteria
- [ ] `cargo test -p common` passes
- [ ] `cargo clippy -p common -- -D warnings` clean
- [ ] All public types are `Send + Sync`
- [ ] `Settings::load()` reads from env vars with `ANKIATLAS_` prefix
- [ ] Validation rejects invalid URLs, out-of-range dimensions, and non-positive integers
- [ ] `get_settings()` returns the same `&'static Settings` on repeated calls
- [ ] `Language` round-trips through `serde_json` (e.g. `"en"` <-> `Language::En`)
- [ ] `configure_logging` produces JSON output when `json_output = true`
- [ ] Correlation ID is propagated across tracing events after `set_correlation_id`
- [ ] All error variants implement `Display` and `std::error::Error`
- [ ] All newtypes (`SlugStr`, `CardId`, `NoteId`, `DeckName`) serialize/deserialize correctly

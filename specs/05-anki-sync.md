# Spec: crate `anki-sync`

## Source Reference
Python: `packages/anki/sync/` (core.py, engine.py, state.py, progress.py, recovery.py)

## Purpose
Orchestrate syncing Anki collection data into PostgreSQL and manage local sync state via SQLite. Provides a high-level `SyncService` that reads an Anki collection, normalizes notes, and upserts everything into PostgreSQL. Also provides a lower-level `SyncEngine` for phased sync operations with progress tracking, an SQLite WAL state database for tracking per-card sync state, and recovery utilities for detecting orphaned or stale cards.

## Dependencies
```toml
[dependencies]
common = { path = "../common" }
database = { path = "../database" }
anki-reader = { path = "../anki-reader" }
sqlx = { version = "0.8", features = ["runtime-tokio", "postgres", "json", "chrono"] }
rusqlite = { version = "0.32", features = ["bundled"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
chrono = { version = "0.4", features = ["serde"] }
tokio = { version = "1", features = ["rt", "sync", "time"] }
tracing = "0.1"
uuid = { version = "1", features = ["v4"] }

[dev-dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
tempfile = "3"
testcontainers = "0.23"
testcontainers-modules = { version = "0.11", features = ["postgres"] }
```

## Public API

### Sync Service (`src/core.rs`)

```rust
use std::path::Path;
use sqlx::PgPool;
use common::error::Result;

/// Statistics from a PostgreSQL sync operation.
#[derive(Debug, Clone, Default)]
pub struct SyncStats {
    pub decks_upserted: i32,
    pub models_upserted: i32,
    pub notes_upserted: i32,
    pub notes_deleted: i32,
    pub cards_upserted: i32,
    pub card_stats_upserted: i32,
    pub duration_ms: i64,
}

/// High-level service for syncing an Anki collection to PostgreSQL.
pub struct SyncService {
    pool: PgPool,
}

impl SyncService {
    pub fn new(pool: PgPool) -> Self;

    /// Read collection from SQLite, normalize notes, upsert all data to PostgreSQL.
    ///
    /// Steps:
    /// 1. Read collection via `anki_reader::read_anki_collection`.
    /// 2. Build deck map and card-deck map.
    /// 3. Normalize all notes (populate `normalized_text`).
    /// 4. In a single transaction:
    ///    a. Upsert decks.
    ///    b. Upsert models.
    ///    c. Upsert notes (set `deleted_at = NULL`).
    ///    d. Soft-delete notes not in collection (set `deleted_at = NOW()`).
    ///    e. Upsert cards.
    ///    f. Upsert card_stats (skip orphaned stats with no matching card).
    ///    g. Update sync_metadata (last_sync_at, last_collection_path).
    /// 5. Return SyncStats with counts and duration.
    pub async fn sync_collection(&self, collection_path: impl AsRef<Path>) -> Result<SyncStats>;
}

/// Convenience function.
pub async fn sync_anki_collection(
    pool: &PgPool,
    collection_path: impl AsRef<Path>,
) -> Result<SyncStats>;
```

### Sync Engine (`src/engine.rs`)

```rust
use crate::progress::ProgressTracker;
use crate::state::StateDB;

/// Result of an engine-level sync operation.
#[derive(Debug, Clone)]
pub struct SyncResult {
    pub cards_created: i32,
    pub cards_updated: i32,
    pub cards_deleted: i32,
    pub cards_skipped: i32,
    pub errors: i32,
    pub duration_ms: i64,
}

/// Phased sync engine with pluggable state and progress tracking.
///
/// Lifecycle: INITIALIZING -> SCANNING -> APPLYING -> COMPLETED/FAILED
pub struct SyncEngine {
    state_db: StateDB,
    progress: ProgressTracker,
    // internal counters
}

impl SyncEngine {
    pub fn new(state_db: StateDB, progress: Option<ProgressTracker>) -> Self;

    /// Access the state database.
    pub fn state_db(&self) -> &StateDB;

    /// Access the progress tracker.
    pub fn progress(&self) -> &ProgressTracker;

    /// Run the sync lifecycle.
    ///
    /// 1. Set phase to SCANNING, count existing states.
    /// 2. If not `dry_run`, set phase to APPLYING and apply changes.
    /// 3. Set phase to COMPLETED (or FAILED on error).
    /// 4. Return SyncResult with counts and duration.
    pub fn sync(&mut self, dry_run: bool) -> Result<SyncResult, Box<dyn std::error::Error>>;
}
```

### State Database (`src/state.rs`)

```rust
use std::path::Path;

/// Tracked state for a single card.
#[derive(Debug, Clone, PartialEq)]
pub struct CardState {
    pub slug: String,
    pub content_hash: String,
    pub anki_guid: Option<i64>,
    pub note_type: String,
    pub source_path: String,
    pub synced_at: f64,
}

/// SQLite WAL database for tracking per-card sync state.
///
/// Schema:
/// ```sql
/// CREATE TABLE IF NOT EXISTS card_state (
///     slug TEXT PRIMARY KEY,
///     content_hash TEXT NOT NULL,
///     anki_guid INTEGER,
///     note_type TEXT NOT NULL DEFAULT '',
///     source_path TEXT NOT NULL DEFAULT '',
///     synced_at REAL NOT NULL DEFAULT 0.0
/// );
/// ```
pub struct StateDB {
    // internal: rusqlite::Connection, path
}

impl StateDB {
    /// Open or create the state database at `db_path`.
    /// Enables WAL mode and foreign keys. Creates the `card_state` table if needed.
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>>;

    /// Get card state by slug, or `None` if not found.
    pub fn get(&self, slug: &str) -> Option<CardState>;

    /// Get all card states, sorted by slug.
    pub fn get_all(&self) -> Vec<CardState>;

    /// Insert or update a card state (upsert on slug).
    pub fn upsert(&self, state: &CardState);

    /// Delete card state by slug.
    pub fn delete(&self, slug: &str);

    /// Get all card states for a given source path, sorted by slug.
    pub fn get_by_source(&self, source_path: &str) -> Vec<CardState>;

    /// Close the database connection.
    pub fn close(self);
}

/// Implement Drop to close the connection automatically.
impl Drop for StateDB { ... }
```

### Progress Tracking (`src/progress.rs`)

```rust
use std::sync::{Arc, Mutex};

/// Phases of a sync operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPhase {
    Initializing,
    Indexing,
    Scanning,
    Generating,
    Applying,
    Completed,
    Failed,
}

impl SyncPhase {
    pub fn as_str(&self) -> &'static str;
}

/// Frozen snapshot of sync progress.
#[derive(Debug, Clone)]
pub struct SyncProgress {
    pub session_id: String,
    pub phase: SyncPhase,
    pub total_notes: i32,
    pub notes_processed: i32,
    pub cards_created: i32,
    pub cards_updated: i32,
    pub cards_deleted: i32,
    pub errors: i32,
    pub started_at: f64,
    pub updated_at: f64,
}

/// Valid stat names for `increment`.
pub const VALID_STATS: &[&str] = &[
    "notes_processed",
    "cards_created",
    "cards_updated",
    "cards_deleted",
    "errors",
];

/// Thread-safe progress tracker.
///
/// All methods acquire an internal `Mutex` before reading/writing.
pub struct ProgressTracker {
    // internal: Arc<Mutex<ProgressState>>
}

impl ProgressTracker {
    /// Create a new tracker with a random or provided session ID.
    pub fn new(session_id: Option<String>) -> Self;

    /// Set the current phase.
    pub fn set_phase(&self, phase: SyncPhase);

    /// Set the total number of notes to process.
    pub fn set_total(&self, total: i32);

    /// Increment a named stat by `count`.
    /// Panics if `stat` is not in VALID_STATS.
    pub fn increment(&self, stat: &str, count: i32);

    /// Get a frozen snapshot of current progress.
    pub fn snapshot(&self) -> SyncProgress;

    /// Mark sync as completed (success=true) or failed (success=false).
    pub fn complete(&self, success: bool);

    /// Get progress percentage (0.0 to 100.0).
    pub fn progress_pct(&self) -> f64;
}

/// ProgressTracker is Clone (wraps Arc internally).
impl Clone for ProgressTracker { ... }
```

### Recovery (`src/recovery.rs`)

```rust
use std::collections::HashSet;
use crate::state::{CardState, StateDB};

/// A recorded action that can be rolled back.
#[derive(Debug, Clone)]
pub struct RollbackAction {
    pub action_type: String,
    pub target_id: String,
    pub succeeded: bool,
    pub error: String,
}

/// Atomic card operation with rollback support.
///
/// Use as a scope guard: if the transaction is dropped without `commit()`,
/// `rollback()` is called automatically.
pub struct CardTransaction {
    // internal: actions list, committed flag
}

impl CardTransaction {
    pub fn new() -> Self;

    /// Record an action for potential rollback.
    pub fn add_rollback(&mut self, action_type: &str, target_id: &str);

    /// Mark as committed -- rollback becomes a no-op.
    pub fn commit(&mut self);

    /// Roll back all recorded actions in reverse order.
    /// Returns the rollback results.
    pub fn rollback(&mut self) -> Vec<RollbackAction>;
}

/// Drop triggers rollback if not committed.
impl Drop for CardTransaction { ... }

/// Detect and recover from inconsistent card states.
pub struct CardRecovery<'a> {
    state_db: &'a StateDB,
}

impl<'a> CardRecovery<'a> {
    pub fn new(state_db: &'a StateDB) -> Self;

    /// Find orphaned cards: in DB but not Anki, and in Anki but not DB.
    /// Returns (in_db_not_anki, in_anki_not_db).
    pub fn find_orphaned(
        &self,
        db_slugs: &HashSet<String>,
        anki_slugs: &HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>);

    /// Find card states older than `max_age_days`.
    /// Only considers states with `synced_at > 0`.
    pub fn find_stale(&self, max_age_days: u32) -> Vec<CardState>;
}
```

### Module root (`src/lib.rs`)

```rust
pub mod core;
pub mod engine;
pub mod progress;
pub mod recovery;
pub mod state;

pub use core::{sync_anki_collection, SyncService, SyncStats};
pub use engine::{SyncEngine, SyncResult};
pub use progress::{ProgressTracker, SyncPhase, SyncProgress};
pub use recovery::{CardRecovery, CardTransaction, RollbackAction};
pub use state::{CardState, StateDB};
```

## Internal Details

### PostgreSQL upsert strategy
All upserts use `INSERT ... ON CONFLICT (pk) DO UPDATE SET ...`. The sync happens within a single transaction. If any upsert fails, the entire transaction rolls back.

### Soft-delete for notes
Notes removed from the Anki collection are not physically deleted from PostgreSQL. Instead, `deleted_at` is set to `NOW()`. The soft-delete query uses `WHERE note_id NOT IN (SELECT unnest($1::bigint[]))` to find notes present in the database but absent from the current collection.

### Card stats filtering
When upserting card_stats, skip any stats whose `card_id` does not exist in the current card set. This handles orphaned revlog entries from deleted cards.

### Sync metadata
Two keys are maintained in `sync_metadata`:
- `last_sync_at`: ISO 8601 timestamp of the last sync as a JSON string.
- `last_collection_path`: path to the last synced collection file as a JSON string.

### StateDB WAL mode
The SQLite state database uses WAL (Write-Ahead Logging) mode for concurrent read access. The `PRAGMA journal_mode=WAL` and `PRAGMA foreign_keys=ON` are set on connection open.

### ProgressTracker thread safety
The `ProgressTracker` wraps all mutable state in `Arc<Mutex<...>>`. This allows it to be shared across threads (e.g., a background sync task and a progress-polling API endpoint). The `Clone` implementation clones the `Arc`, sharing the same underlying state.

### CardTransaction Drop behavior
When a `CardTransaction` is dropped without calling `commit()`, `rollback()` is invoked automatically. This is a RAII guard pattern. After `commit()`, `rollback()` returns an empty vec.

### Recovery stale detection
`find_stale` computes a cutoff as `current_time - (max_age_days * 86400)` and returns all states where `0 < synced_at < cutoff`. The `synced_at == 0` case is excluded because it indicates a state that was never synced.

## Acceptance Criteria
- [ ] `cargo test -p anki-sync` passes
- [ ] `cargo clippy -p anki-sync -- -D warnings` clean
- [ ] All public types are `Send + Sync`
- [ ] `SyncService::sync_collection` reads a test `.anki2` file and upserts all data into PostgreSQL
- [ ] Upserts are idempotent -- syncing the same collection twice produces the same database state
- [ ] Soft-delete marks notes not in the collection with `deleted_at`
- [ ] Card stats with no matching card are skipped during sync
- [ ] `SyncStats` correctly reports counts for all entity types and duration
- [ ] `StateDB::open` creates the database file and `card_state` table
- [ ] `StateDB::upsert` inserts new states and updates existing ones (by slug)
- [ ] `StateDB::get_all` returns states sorted by slug
- [ ] `StateDB::get_by_source` filters by source_path
- [ ] `StateDB::delete` removes a state by slug
- [ ] `StateDB` uses WAL journal mode (verify with PRAGMA query in test)
- [ ] `ProgressTracker` is thread-safe (spawn two threads, increment concurrently, verify totals)
- [ ] `ProgressTracker::set_phase` updates the phase and `updated_at`
- [ ] `ProgressTracker::progress_pct` returns 0.0 when total is 0, correct percentage otherwise
- [ ] `ProgressTracker::increment` panics on invalid stat names
- [ ] `SyncEngine::sync` transitions through SCANNING -> APPLYING -> COMPLETED
- [ ] `SyncEngine::sync` with `dry_run=true` does not enter APPLYING phase
- [ ] `SyncEngine::sync` sets phase to FAILED on error
- [ ] `CardTransaction::commit` prevents rollback
- [ ] `CardTransaction` drop without commit triggers rollback
- [ ] `CardRecovery::find_orphaned` correctly computes set differences
- [ ] `CardRecovery::find_stale` returns only states with `0 < synced_at < cutoff`
- [ ] `CardRecovery::find_stale` excludes states with `synced_at == 0`

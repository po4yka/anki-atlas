# Spec: crate `database`

## Source Reference
Python: `packages/common/database.py` + `packages/common/migrations/*.sql`

## Purpose
Async PostgreSQL database layer providing connection pool management, migration execution, and health checks. Uses `sqlx` with compile-time query verification disabled (runtime mode) since the schema is applied via migrations. Embeds SQL migration files and applies them idempotently using a `schema_migrations` tracking table.

## Dependencies
```toml
[dependencies]
common = { path = "../common" }
sqlx = { version = "0.8", features = ["runtime-tokio", "tls-rustls", "postgres", "json", "chrono"] }
tokio = { version = "1", features = ["rt", "sync"] }
tracing = "0.1"
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

[dev-dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
testcontainers = "0.23"
testcontainers-modules = { version = "0.11", features = ["postgres"] }
```

## Public API

### Pool Management (`src/pool.rs`)

```rust
use sqlx::PgPool;
use common::config::Settings;
use common::error::Result;

/// Create a new PgPool from settings.
///
/// Pool configuration:
/// - min_connections: 2
/// - max_connections: 10
/// - acquire_timeout: 10 seconds
pub async fn create_pool(settings: &Settings) -> Result<PgPool>;

/// Check if the database is reachable by executing `SELECT 1`.
/// Returns `false` on any error (connection timeout = 5s).
pub async fn check_connection(pool: &PgPool) -> bool;
```

### Connection Context Manager (`src/connection.rs`)

```rust
use sqlx::{PgPool, Postgres, Transaction};
use common::error::Result;

/// Acquire a connection from the pool, execute a closure, and return the result.
/// The connection is automatically returned to the pool.
pub async fn with_connection<F, T>(pool: &PgPool, f: F) -> Result<T>
where
    F: for<'c> FnOnce(&'c mut sqlx::PgConnection) -> futures::future::BoxFuture<'c, Result<T>>;

/// Begin a transaction, execute a closure, and commit on success / rollback on error.
pub async fn with_transaction<F, T>(pool: &PgPool, f: F) -> Result<T>
where
    F: for<'c> FnOnce(&'c mut Transaction<'_, Postgres>) -> futures::future::BoxFuture<'c, Result<T>>;
```

### Migrations (`src/migrations.rs`)

```rust
use sqlx::PgPool;
use common::error::Result;

/// Result of running migrations.
#[derive(Debug, Clone, Default)]
pub struct MigrationResult {
    /// Names of migrations that were applied in this run.
    pub applied: Vec<String>,
    /// Names of migrations that were already applied (skipped).
    pub skipped: Vec<String>,
}

/// Run all pending SQL migrations.
///
/// Steps:
/// 1. Create `schema_migrations` table if it does not exist.
/// 2. Load the set of already-applied migration names.
/// 3. For each embedded `.sql` file (sorted by filename):
///    - If name is in the applied set, add to `skipped`.
///    - Otherwise, execute the SQL and insert into `schema_migrations`.
/// 4. Each migration runs in its own transaction (applied atomically).
///
/// Migration files are embedded at compile time using `include_str!`.
pub async fn run_migrations(pool: &PgPool) -> Result<MigrationResult>;
```

### Embedded Migrations

The following SQL files are embedded via `include_str!` in a static array:

```rust
/// (name, sql) pairs, sorted by name.
static MIGRATIONS: &[(&str, &str)] = &[
    ("001_initial_schema", include_str!("../migrations/001_initial_schema.sql")),
    ("002_pg_trgm_lexical_search", include_str!("../migrations/002_pg_trgm_lexical_search.sql")),
];
```

New migrations are added by:
1. Creating a new `.sql` file in the `migrations/` directory.
2. Adding the corresponding entry to the `MIGRATIONS` array.

### Module root (`src/lib.rs`)

```rust
pub mod connection;
pub mod migrations;
pub mod pool;

pub use migrations::{run_migrations, MigrationResult};
pub use pool::{check_connection, create_pool};
```

## Internal Details

### Migration tracking table
```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```
This table is created by `run_migrations` on first call -- it does not require a migration file itself.

### Migration atomicity
Each migration file runs inside a single transaction. If the SQL fails, the transaction rolls back and the migration name is NOT inserted into `schema_migrations`. The function returns an error immediately (does not continue to subsequent migrations).

Exception: migrations that contain statements incompatible with transactions (e.g. `CREATE INDEX CONCURRENTLY`) should be detected and run outside a transaction. For now, the two existing migrations are transaction-safe.

### Pool lifecycle
The crate does NOT manage a global singleton pool. Callers (typically the app entry point) create the pool, pass it to services, and drop it on shutdown. This avoids the need for `once_cell` or hidden global state.

### Connection URL parsing
The `postgres_url` from `Settings` is passed directly to `sqlx::PgPool::connect_with`. The URL validation in the `common` crate ensures it starts with `postgresql://` or `postgres://`.

### Migrations directory
Copy the two SQL files from `packages/common/migrations/` into the Rust crate at `crates/database/migrations/`. The content must be identical to the Python source.

## Acceptance Criteria
- [ ] `cargo test -p database` passes (requires testcontainers with Docker)
- [ ] `cargo clippy -p database -- -D warnings` clean
- [ ] All public types are `Send + Sync`
- [ ] `create_pool` connects to PostgreSQL and returns a usable pool
- [ ] `check_connection` returns `true` for a live database, `false` when unreachable
- [ ] `run_migrations` creates `schema_migrations` table on first run
- [ ] `run_migrations` applies `001_initial_schema` creating all tables (decks, notes, cards, card_stats, models, sync_metadata, topics, note_topics)
- [ ] `run_migrations` applies `002_pg_trgm_lexical_search` creating trigram indexes
- [ ] `run_migrations` is idempotent -- second call applies nothing, skips all
- [ ] `run_migrations` returns correct `applied` and `skipped` lists
- [ ] A failed migration rolls back its transaction and returns an error
- [ ] `with_transaction` commits on success and rolls back on error
- [ ] The embedded SQL files are byte-identical to the Python source migration files

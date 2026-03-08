use std::collections::HashSet;

use common::error::Result;
use sqlx::PgPool;
use tracing::{info, instrument};

use crate::migration_error;

/// Result of running migrations.
#[derive(Debug, Clone, Default)]
pub struct MigrationResult {
    /// Names of migrations that were applied in this run.
    pub applied: Vec<String>,
    /// Names of migrations that were already applied (skipped).
    pub skipped: Vec<String>,
}

/// (name, sql) pairs, sorted by name.
pub static MIGRATIONS: &[(&str, &str)] = &[
    (
        "001_initial_schema",
        include_str!("../migrations/001_initial_schema.sql"),
    ),
    (
        "002_pg_trgm_lexical_search",
        include_str!("../migrations/002_pg_trgm_lexical_search.sql"),
    ),
];

/// Run all pending SQL migrations.
///
/// Steps:
/// 1. Create `schema_migrations` table if it does not exist.
/// 2. Load the set of already-applied migration names.
/// 3. For each embedded `.sql` file (sorted by filename):
///    - If name is in the applied set, add to `skipped`.
///    - Otherwise, execute the SQL and insert into `schema_migrations`.
/// 4. Each migration runs in its own transaction (applied atomically).
#[instrument(skip_all)]
pub async fn run_migrations(pool: &PgPool) -> Result<MigrationResult> {
    // 1. Create tracking table
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )",
    )
    .execute(pool)
    .await
    .map_err(migration_error("failed to create schema_migrations table"))?;

    // 2. Load already-applied migrations
    let applied_rows: Vec<String> = sqlx::query_scalar("SELECT name FROM schema_migrations")
        .fetch_all(pool)
        .await
        .map_err(migration_error("failed to query schema_migrations"))?;
    let already_applied: HashSet<&str> = applied_rows.iter().map(|s| s.as_str()).collect();

    // 3. Apply each migration
    let mut result = MigrationResult::default();

    for (name, sql) in MIGRATIONS {
        if already_applied.contains(name) {
            info!(migration = name, "skipping already-applied migration");
            result.skipped.push(name.to_string());
            continue;
        }

        info!(migration = name, "applying migration");

        let mut txn = pool
            .begin()
            .await
            .map_err(migration_error(&format!("failed to begin transaction for {name}")))?;

        sqlx::query(sql)
            .execute(&mut *txn)
            .await
            .map_err(migration_error(&format!("migration {name} failed")))?;

        sqlx::query("INSERT INTO schema_migrations (name) VALUES ($1)")
            .bind(name)
            .execute(&mut *txn)
            .await
            .map_err(migration_error(&format!("failed to record migration {name}")))?;

        txn.commit()
            .await
            .map_err(migration_error(&format!("failed to commit migration {name}")))?;

        result.applied.push(name.to_string());
    }

    Ok(result)
}

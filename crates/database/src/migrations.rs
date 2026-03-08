use common::error::Result;
use sqlx::PgPool;

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
pub async fn run_migrations(_pool: &PgPool) -> Result<MigrationResult> {
    todo!()
}

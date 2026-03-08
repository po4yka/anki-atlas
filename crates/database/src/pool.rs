use common::config::Settings;
use common::error::Result;
use sqlx::PgPool;

/// Create a new PgPool from settings.
pub async fn create_pool(_settings: &Settings) -> Result<PgPool> {
    todo!()
}

/// Check if the database is reachable by executing `SELECT 1`.
pub async fn check_connection(_pool: &PgPool) -> bool {
    todo!()
}

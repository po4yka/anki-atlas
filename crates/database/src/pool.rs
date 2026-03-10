use std::time::Duration;

use common::config::DatabaseSettings;
use common::error::Result;
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use tracing::instrument;

use crate::connection_error;

/// Create a new PgPool from settings.
///
/// Pool configuration:
/// - min_connections: 2
/// - max_connections: 10
/// - acquire_timeout: 10 seconds
#[instrument(skip_all)]
pub async fn create_pool(settings: &DatabaseSettings) -> Result<PgPool> {
    PgPoolOptions::new()
        .min_connections(2)
        .max_connections(10)
        .acquire_timeout(Duration::from_secs(10))
        .connect(&settings.postgres_url)
        .await
        .map_err(connection_error)
}

/// Check if the database is reachable by executing `SELECT 1`.
/// Returns `false` on any error (connection timeout = 5s).
#[instrument(skip_all)]
pub async fn check_connection(pool: &PgPool) -> bool {
    tokio::time::timeout(
        Duration::from_secs(5),
        sqlx::query("SELECT 1").execute(pool),
    )
    .await
    .map(|r| r.is_ok())
    .unwrap_or(false)
}

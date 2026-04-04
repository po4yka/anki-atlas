use crate::error::JobError;
use sqlx::PgPool;

/// Create a PostgreSQL connection pool for the job queue.
///
/// Reuses the application's existing PostgreSQL database -- no additional
/// infrastructure required.
pub async fn create_job_pool(postgres_url: &str) -> Result<PgPool, JobError> {
    sqlx::postgres::PgPoolOptions::new()
        .max_connections(5)
        .connect(postgres_url)
        .await
        .map_err(|e| JobError::BackendUnavailable(format!("postgres connection failed: {e}")))
}

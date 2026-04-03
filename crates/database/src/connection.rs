use common::error::Result;
use sqlx::{PgPool, Postgres, Transaction};
use tracing::{instrument, warn};

use crate::connection_error;

/// Acquire a connection from the pool, execute a closure, and return the result.
/// The connection is automatically returned to the pool.
#[instrument(skip_all)]
pub async fn with_connection<F, T>(pool: &PgPool, f: F) -> Result<T>
where
    F: for<'c> FnOnce(&'c mut sqlx::PgConnection) -> futures::future::BoxFuture<'c, Result<T>>,
{
    let mut conn = pool.acquire().await.map_err(connection_error)?;
    f(conn.as_mut()).await
}

/// Begin a transaction, execute a closure, and commit on success / rollback on error.
#[instrument(skip_all)]
pub async fn with_transaction<F, T>(pool: &PgPool, f: F) -> Result<T>
where
    F: for<'c> FnOnce(
        &'c mut Transaction<'_, Postgres>,
    ) -> futures::future::BoxFuture<'c, Result<T>>,
{
    let mut txn = pool.begin().await.map_err(connection_error)?;

    match f(&mut txn).await {
        Ok(value) => {
            txn.commit().await.map_err(connection_error)?;
            Ok(value)
        }
        Err(e) => {
            if let Err(rb_err) = txn.rollback().await {
                warn!(%rb_err, "explicit rollback failed (will rollback on drop)");
            }
            Err(e)
        }
    }
}

use std::collections::HashMap;

use common::error::{AnkiAtlasError, Result};
use sqlx::{PgPool, Postgres, Transaction};
use tracing::instrument;

/// Acquire a connection from the pool, execute a closure, and return the result.
/// The connection is automatically returned to the pool.
#[instrument(skip_all)]
pub async fn with_connection<F, T>(pool: &PgPool, f: F) -> Result<T>
where
    F: for<'c> FnOnce(&'c mut sqlx::PgConnection) -> futures::future::BoxFuture<'c, Result<T>>,
{
    let mut conn = pool
        .acquire()
        .await
        .map_err(|e| AnkiAtlasError::DatabaseConnection {
            message: e.to_string(),
            context: HashMap::new(),
        })?;
    f(conn.as_mut()).await
}

/// Begin a transaction, execute a closure, and commit on success / rollback on error.
#[instrument(skip_all)]
pub async fn with_transaction<F, T>(pool: &PgPool, f: F) -> Result<T>
where
    F: for<'c> FnOnce(&'c mut Transaction<'_, Postgres>) -> futures::future::BoxFuture<'c, Result<T>>,
{
    let mut txn = pool
        .begin()
        .await
        .map_err(|e| AnkiAtlasError::DatabaseConnection {
            message: e.to_string(),
            context: HashMap::new(),
        })?;

    match f(&mut txn).await {
        Ok(value) => {
            txn.commit()
                .await
                .map_err(|e| AnkiAtlasError::DatabaseConnection {
                    message: e.to_string(),
                    context: HashMap::new(),
                })?;
            Ok(value)
        }
        Err(e) => {
            // Rollback is implicit on drop, but explicit is clearer
            let _ = txn.rollback().await;
            Err(e)
        }
    }
}

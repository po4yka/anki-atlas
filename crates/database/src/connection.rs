use common::error::Result;
use sqlx::{PgPool, Postgres, Transaction};

/// Acquire a connection from the pool, execute a closure, and return the result.
pub async fn with_connection<F, T>(_pool: &PgPool, _f: F) -> Result<T>
where
    F: for<'c> FnOnce(&'c mut sqlx::PgConnection) -> futures::future::BoxFuture<'c, Result<T>>,
{
    todo!()
}

/// Begin a transaction, execute a closure, and commit on success / rollback on error.
pub async fn with_transaction<F, T>(_pool: &PgPool, _f: F) -> Result<T>
where
    F: for<'c> FnOnce(&'c mut Transaction<'_, Postgres>) -> futures::future::BoxFuture<'c, Result<T>>,
{
    todo!()
}

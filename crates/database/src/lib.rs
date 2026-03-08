pub mod connection;
pub mod migrations;
pub mod pool;

pub use migrations::{run_migrations, MigrationResult};
pub use pool::{check_connection, create_pool};

use std::collections::HashMap;

use common::error::AnkiAtlasError;

/// Convert a sqlx error into a `DatabaseConnection` error.
fn connection_error(e: sqlx::Error) -> AnkiAtlasError {
    AnkiAtlasError::DatabaseConnection {
        message: e.to_string(),
        context: HashMap::new(),
    }
}

/// Create a mapper that converts a sqlx error into a `Migration` error with context.
fn migration_error(context_msg: &str) -> impl FnOnce(sqlx::Error) -> AnkiAtlasError + '_ {
    move |e| AnkiAtlasError::Migration {
        message: format!("{context_msg}: {e}"),
        context: HashMap::new(),
    }
}

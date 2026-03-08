#![allow(dead_code, unused_variables)]

use std::path::Path;

use sqlx::PgPool;

use common::error::Result;

/// Statistics from a PostgreSQL sync operation.
#[derive(Debug, Clone, Default)]
pub struct SyncStats {
    pub decks_upserted: i32,
    pub models_upserted: i32,
    pub notes_upserted: i32,
    pub notes_deleted: i32,
    pub cards_upserted: i32,
    pub card_stats_upserted: i32,
    pub duration_ms: i64,
}

/// High-level service for syncing an Anki collection to PostgreSQL.
pub struct SyncService {
    pool: PgPool,
}

impl SyncService {
    pub fn new(pool: PgPool) -> Self {
        todo!()
    }

    /// Read collection from SQLite, normalize notes, upsert all data to PostgreSQL.
    pub async fn sync_collection(&self, collection_path: impl AsRef<Path>) -> Result<SyncStats> {
        todo!()
    }
}

/// Convenience function.
pub async fn sync_anki_collection(
    pool: &PgPool,
    collection_path: impl AsRef<Path>,
) -> Result<SyncStats> {
    todo!()
}

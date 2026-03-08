use anki_sync::{SyncService, SyncStats, sync_anki_collection};
use rusqlite::Connection;
use sqlx::PgPool;
use tempfile::NamedTempFile;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a minimal legacy Anki SQLite database with 2 decks, 1 model, 2 notes,
/// 2 cards, and 3 revlog entries (matching card 500 only).
fn create_test_anki_db() -> NamedTempFile {
    let file = NamedTempFile::new().expect("create temp file");
    let conn = Connection::open(file.path()).expect("open sqlite");

    conn.execute_batch(
        "
        CREATE TABLE col (
            id INTEGER PRIMARY KEY,
            crt INTEGER, mod INTEGER, scm INTEGER, ver INTEGER,
            dty INTEGER, usn INTEGER, ls INTEGER,
            conf TEXT, models TEXT, decks TEXT, dconf TEXT, tags TEXT
        );
        INSERT INTO col VALUES (
            1, 0, 0, 0, 11, 0, 0, 0, '{}',
            '{\"1234567890\": {\"id\": 1234567890, \"name\": \"Basic\", \"flds\": [{\"name\": \"Front\", \"ord\": 0}, {\"name\": \"Back\", \"ord\": 1}], \"tmpls\": [{\"name\": \"Card 1\"}]}}',
            '{\"1\": {\"id\": 1, \"name\": \"Default\"}, \"2\": {\"id\": 2, \"name\": \"Japanese::Vocab\"}}',
            '{}', '{}'
        );

        CREATE TABLE notes (
            id INTEGER PRIMARY KEY, guid TEXT, mid INTEGER, mod INTEGER,
            usn INTEGER, tags TEXT, flds TEXT, sfld TEXT, csum INTEGER,
            flags INTEGER, data TEXT
        );
        INSERT INTO notes VALUES (100, 'abc', 1234567890, 1700000000, -1, ' vocab ', 'Hello\x1fWorld', 'Hello', 0, 0, '');
        INSERT INTO notes VALUES (101, 'def', 1234567890, 1700000001, -1, '', 'Question\x1fAnswer', 'Question', 0, 0, '');

        CREATE TABLE cards (
            id INTEGER PRIMARY KEY, nid INTEGER, did INTEGER, ord INTEGER,
            mod INTEGER, usn INTEGER, type INTEGER, queue INTEGER,
            due INTEGER, ivl INTEGER, factor INTEGER, reps INTEGER,
            lapses INTEGER, left INTEGER, odue INTEGER, odid INTEGER,
            flags INTEGER, data TEXT
        );
        INSERT INTO cards VALUES (500, 100, 1, 0, 1700000000, -1, 2, 2, 1000, 30, 2500, 10, 2, 0, 0, 0, 0, '');
        INSERT INTO cards VALUES (501, 101, 2, 0, 1700000001, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '');

        CREATE TABLE revlog (
            id INTEGER PRIMARY KEY, cid INTEGER, usn INTEGER, ease INTEGER,
            ivl INTEGER, lastIvl INTEGER, factor INTEGER, time INTEGER, type INTEGER
        );
        INSERT INTO revlog VALUES (1700000000000, 500, -1, 3, 30, 15, 2500, 8000, 1);
        INSERT INTO revlog VALUES (1700000001000, 500, -1, 1, 0, 30, 2100, 15000, 2);
        INSERT INTO revlog VALUES (1700000002000, 500, -1, 3, 10, 0, 2200, 6000, 1);
        ",
    )
    .expect("create test db");

    file
}

/// Create a test database with only 1 note and 1 card (subset of the full DB).
/// Used for soft-delete testing: sync this AFTER syncing the full DB.
fn create_subset_anki_db() -> NamedTempFile {
    let file = NamedTempFile::new().expect("create temp file");
    let conn = Connection::open(file.path()).expect("open sqlite");

    conn.execute_batch(
        "
        CREATE TABLE col (
            id INTEGER PRIMARY KEY,
            crt INTEGER, mod INTEGER, scm INTEGER, ver INTEGER,
            dty INTEGER, usn INTEGER, ls INTEGER,
            conf TEXT, models TEXT, decks TEXT, dconf TEXT, tags TEXT
        );
        INSERT INTO col VALUES (
            1, 0, 0, 0, 11, 0, 0, 0, '{}',
            '{\"1234567890\": {\"id\": 1234567890, \"name\": \"Basic\", \"flds\": [{\"name\": \"Front\", \"ord\": 0}, {\"name\": \"Back\", \"ord\": 1}], \"tmpls\": [{\"name\": \"Card 1\"}]}}',
            '{\"1\": {\"id\": 1, \"name\": \"Default\"}}',
            '{}', '{}'
        );

        CREATE TABLE notes (
            id INTEGER PRIMARY KEY, guid TEXT, mid INTEGER, mod INTEGER,
            usn INTEGER, tags TEXT, flds TEXT, sfld TEXT, csum INTEGER,
            flags INTEGER, data TEXT
        );
        INSERT INTO notes VALUES (100, 'abc', 1234567890, 1700000000, -1, ' vocab ', 'Hello\x1fWorld', 'Hello', 0, 0, '');

        CREATE TABLE cards (
            id INTEGER PRIMARY KEY, nid INTEGER, did INTEGER, ord INTEGER,
            mod INTEGER, usn INTEGER, type INTEGER, queue INTEGER,
            due INTEGER, ivl INTEGER, factor INTEGER, reps INTEGER,
            lapses INTEGER, left INTEGER, odue INTEGER, odid INTEGER,
            flags INTEGER, data TEXT
        );
        INSERT INTO cards VALUES (500, 100, 1, 0, 1700000000, -1, 2, 2, 1000, 30, 2500, 10, 2, 0, 0, 0, 0, '');

        CREATE TABLE revlog (
            id INTEGER PRIMARY KEY, cid INTEGER, usn INTEGER, ease INTEGER,
            ivl INTEGER, lastIvl INTEGER, factor INTEGER, time INTEGER, type INTEGER
        );
        INSERT INTO revlog VALUES (1700000000000, 500, -1, 3, 30, 15, 2500, 8000, 1);
        ",
    )
    .expect("create subset db");

    file
}

/// Create an Anki DB with a card_stats entry for a card that does NOT exist
/// in the cards table (orphaned revlog).
fn create_orphan_stats_anki_db() -> NamedTempFile {
    let file = NamedTempFile::new().expect("create temp file");
    let conn = Connection::open(file.path()).expect("open sqlite");

    conn.execute_batch(
        "
        CREATE TABLE col (
            id INTEGER PRIMARY KEY,
            crt INTEGER, mod INTEGER, scm INTEGER, ver INTEGER,
            dty INTEGER, usn INTEGER, ls INTEGER,
            conf TEXT, models TEXT, decks TEXT, dconf TEXT, tags TEXT
        );
        INSERT INTO col VALUES (
            1, 0, 0, 0, 11, 0, 0, 0, '{}',
            '{\"1234567890\": {\"id\": 1234567890, \"name\": \"Basic\", \"flds\": [{\"name\": \"Front\", \"ord\": 0}, {\"name\": \"Back\", \"ord\": 1}], \"tmpls\": [{\"name\": \"Card 1\"}]}}',
            '{\"1\": {\"id\": 1, \"name\": \"Default\"}}',
            '{}', '{}'
        );

        CREATE TABLE notes (
            id INTEGER PRIMARY KEY, guid TEXT, mid INTEGER, mod INTEGER,
            usn INTEGER, tags TEXT, flds TEXT, sfld TEXT, csum INTEGER,
            flags INTEGER, data TEXT
        );
        INSERT INTO notes VALUES (100, 'abc', 1234567890, 1700000000, -1, '', 'Hello\x1fWorld', 'Hello', 0, 0, '');

        CREATE TABLE cards (
            id INTEGER PRIMARY KEY, nid INTEGER, did INTEGER, ord INTEGER,
            mod INTEGER, usn INTEGER, type INTEGER, queue INTEGER,
            due INTEGER, ivl INTEGER, factor INTEGER, reps INTEGER,
            lapses INTEGER, left INTEGER, odue INTEGER, odid INTEGER,
            flags INTEGER, data TEXT
        );
        INSERT INTO cards VALUES (500, 100, 1, 0, 1700000000, -1, 2, 2, 1000, 30, 2500, 10, 2, 0, 0, 0, 0, '');

        CREATE TABLE revlog (
            id INTEGER PRIMARY KEY, cid INTEGER, usn INTEGER, ease INTEGER,
            ivl INTEGER, lastIvl INTEGER, factor INTEGER, time INTEGER, type INTEGER
        );
        -- Revlog for existing card 500
        INSERT INTO revlog VALUES (1700000000000, 500, -1, 3, 30, 15, 2500, 8000, 1);
        -- Revlog for DELETED card 999 (orphaned)
        INSERT INTO revlog VALUES (1700000001000, 999, -1, 2, 10, 5, 2000, 3000, 1);
        INSERT INTO revlog VALUES (1700000002000, 999, -1, 3, 20, 10, 2200, 4000, 1);
        ",
    )
    .expect("create orphan stats db");

    file
}

/// Start a PostgreSQL testcontainer, run migrations, and return the pool.
async fn setup_pg() -> (PgPool, testcontainers::ContainerAsync<Postgres>) {
    let container = Postgres::default()
        .start()
        .await
        .expect("start postgres container");

    let host_port = container
        .get_host_port_ipv4(5432)
        .await
        .expect("get port");

    let url = format!("postgresql://postgres:postgres@127.0.0.1:{host_port}/postgres");
    let pool = PgPool::connect(&url).await.expect("connect to pg");

    // Run migrations to create schema
    database::run_migrations(&pool)
        .await
        .expect("run migrations");

    (pool, container)
}

// ---------------------------------------------------------------------------
// SyncStats tests (no dependencies needed)
// ---------------------------------------------------------------------------

#[test]
fn sync_stats_default_is_all_zeros() {
    let stats = SyncStats::default();
    assert_eq!(stats.decks_upserted, 0);
    assert_eq!(stats.models_upserted, 0);
    assert_eq!(stats.notes_upserted, 0);
    assert_eq!(stats.notes_deleted, 0);
    assert_eq!(stats.cards_upserted, 0);
    assert_eq!(stats.card_stats_upserted, 0);
    assert_eq!(stats.duration_ms, 0);
}

#[test]
fn sync_stats_debug_impl() {
    let stats = SyncStats::default();
    let debug = format!("{stats:?}");
    assert!(debug.contains("SyncStats"));
}

#[test]
fn sync_stats_clone() {
    let mut stats = SyncStats::default();
    stats.decks_upserted = 5;
    let cloned = stats.clone();
    assert_eq!(cloned.decks_upserted, 5);
}

// ---------------------------------------------------------------------------
// SyncService::new
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sync_service_new_creates_instance() {
    let (pool, _container) = setup_pg().await;
    let _service = SyncService::new(pool);
    // Should not panic -- construction succeeds
}

// ---------------------------------------------------------------------------
// SyncService::sync_collection -- full sync
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sync_collection_upserts_decks() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();
    let service = SyncService::new(pool.clone());

    let stats = service.sync_collection(db.path()).await.unwrap();
    assert_eq!(stats.decks_upserted, 2);

    // Verify decks exist in PostgreSQL
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM decks")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count.0, 2);
}

#[tokio::test]
async fn sync_collection_upserts_models() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();
    let service = SyncService::new(pool.clone());

    let stats = service.sync_collection(db.path()).await.unwrap();
    assert_eq!(stats.models_upserted, 1);

    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM models")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count.0, 1);
}

#[tokio::test]
async fn sync_collection_upserts_notes() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();
    let service = SyncService::new(pool.clone());

    let stats = service.sync_collection(db.path()).await.unwrap();
    assert_eq!(stats.notes_upserted, 2);

    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count.0, 2);
}

#[tokio::test]
async fn sync_collection_upserts_cards() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();
    let service = SyncService::new(pool.clone());

    let stats = service.sync_collection(db.path()).await.unwrap();
    assert_eq!(stats.cards_upserted, 2);

    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM cards")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count.0, 2);
}

#[tokio::test]
async fn sync_collection_upserts_card_stats() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();
    let service = SyncService::new(pool.clone());

    let stats = service.sync_collection(db.path()).await.unwrap();
    // Only card 500 has revlog entries
    assert_eq!(stats.card_stats_upserted, 1);

    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM card_stats")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count.0, 1);
}

#[tokio::test]
async fn sync_collection_reports_correct_stats() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();
    let service = SyncService::new(pool);

    let stats = service.sync_collection(db.path()).await.unwrap();

    assert_eq!(stats.decks_upserted, 2);
    assert_eq!(stats.models_upserted, 1);
    assert_eq!(stats.notes_upserted, 2);
    assert_eq!(stats.notes_deleted, 0); // No pre-existing notes to delete
    assert_eq!(stats.cards_upserted, 2);
    assert_eq!(stats.card_stats_upserted, 1);
    assert!(stats.duration_ms >= 0);
}

#[tokio::test]
async fn sync_collection_duration_is_non_negative() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();
    let service = SyncService::new(pool);

    let stats = service.sync_collection(db.path()).await.unwrap();
    assert!(stats.duration_ms >= 0);
}

// ---------------------------------------------------------------------------
// Idempotency
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sync_collection_is_idempotent() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();
    let service = SyncService::new(pool.clone());

    // Sync twice with the same collection
    let stats1 = service.sync_collection(db.path()).await.unwrap();
    let stats2 = service.sync_collection(db.path()).await.unwrap();

    // Counts should be the same (upserts are idempotent)
    assert_eq!(stats1.decks_upserted, stats2.decks_upserted);
    assert_eq!(stats1.models_upserted, stats2.models_upserted);
    assert_eq!(stats1.notes_upserted, stats2.notes_upserted);
    assert_eq!(stats1.cards_upserted, stats2.cards_upserted);
    assert_eq!(stats1.card_stats_upserted, stats2.card_stats_upserted);

    // Database should have same row counts as after first sync
    let deck_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM decks")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(deck_count.0, 2);

    let note_count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(note_count.0, 2);
}

// ---------------------------------------------------------------------------
// Soft-delete
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sync_collection_soft_deletes_missing_notes() {
    let (pool, _container) = setup_pg().await;
    let service = SyncService::new(pool.clone());

    // First sync: 2 notes
    let full_db = create_test_anki_db();
    service.sync_collection(full_db.path()).await.unwrap();

    // Second sync: only 1 note (note 101 is removed)
    let subset_db = create_subset_anki_db();
    let stats = service.sync_collection(subset_db.path()).await.unwrap();

    // Note 101 should be soft-deleted
    assert_eq!(stats.notes_deleted, 1);

    // Verify: 1 note with deleted_at IS NULL, 1 with deleted_at IS NOT NULL
    let active: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(active.0, 1);

    let deleted: (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM notes WHERE deleted_at IS NOT NULL")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(deleted.0, 1);
}

#[tokio::test]
async fn sync_collection_clears_deleted_at_on_re_add() {
    let (pool, _container) = setup_pg().await;
    let service = SyncService::new(pool.clone());

    // Sync full (2 notes), then subset (1 note) to soft-delete note 101
    let full_db = create_test_anki_db();
    service.sync_collection(full_db.path()).await.unwrap();

    let subset_db = create_subset_anki_db();
    service.sync_collection(subset_db.path()).await.unwrap();

    // Re-sync full: note 101 should be restored (deleted_at = NULL)
    let full_db2 = create_test_anki_db();
    service.sync_collection(full_db2.path()).await.unwrap();

    let active: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(active.0, 2);
}

// ---------------------------------------------------------------------------
// Card stats filtering (orphaned revlog)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sync_collection_skips_orphaned_card_stats() {
    let (pool, _container) = setup_pg().await;
    let db = create_orphan_stats_anki_db();
    let service = SyncService::new(pool.clone());

    let stats = service.sync_collection(db.path()).await.unwrap();

    // Only card 500 stats should be upserted; card 999 stats are orphaned
    assert_eq!(stats.card_stats_upserted, 1);

    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM card_stats")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count.0, 1);
}

// ---------------------------------------------------------------------------
// Sync metadata
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sync_collection_updates_sync_metadata() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();
    let service = SyncService::new(pool.clone());

    service.sync_collection(db.path()).await.unwrap();

    // Check last_sync_at exists
    let row: (serde_json::Value,) =
        sqlx::query_as("SELECT value FROM sync_metadata WHERE key = 'last_sync_at'")
            .fetch_one(&pool)
            .await
            .unwrap();
    // Value should be a JSON string with ISO 8601 timestamp
    let ts = row.0.as_str().expect("last_sync_at should be a string");
    assert!(!ts.is_empty());

    // Check last_collection_path exists
    let row: (serde_json::Value,) =
        sqlx::query_as("SELECT value FROM sync_metadata WHERE key = 'last_collection_path'")
            .fetch_one(&pool)
            .await
            .unwrap();
    let path = row.0.as_str().expect("last_collection_path should be a string");
    assert!(!path.is_empty());
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sync_collection_nonexistent_file_returns_error() {
    let (pool, _container) = setup_pg().await;
    let service = SyncService::new(pool);

    let result = service
        .sync_collection("/nonexistent/collection.anki2")
        .await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Data integrity
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sync_collection_stores_correct_deck_data() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();
    let service = SyncService::new(pool.clone());

    service.sync_collection(db.path()).await.unwrap();

    let row: (i64, String) =
        sqlx::query_as("SELECT deck_id, name FROM decks WHERE deck_id = 1")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(row.0, 1);
    assert_eq!(row.1, "Default");
}

#[tokio::test]
async fn sync_collection_stores_correct_note_data() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();
    let service = SyncService::new(pool.clone());

    service.sync_collection(db.path()).await.unwrap();

    let row: (i64, i64) =
        sqlx::query_as("SELECT note_id, model_id FROM notes WHERE note_id = 100")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(row.0, 100);
    assert_eq!(row.1, 1234567890);
}

#[tokio::test]
async fn sync_collection_stores_correct_card_data() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();
    let service = SyncService::new(pool.clone());

    service.sync_collection(db.path()).await.unwrap();

    let row: (i64, i64, i64) =
        sqlx::query_as("SELECT card_id, note_id, deck_id FROM cards WHERE card_id = 500")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(row.0, 500);
    assert_eq!(row.1, 100);
    assert_eq!(row.2, 1);
}

#[tokio::test]
async fn sync_collection_stores_correct_card_stats_data() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();
    let service = SyncService::new(pool.clone());

    service.sync_collection(db.path()).await.unwrap();

    let row: (i64, i32, i64) =
        sqlx::query_as("SELECT card_id, reviews, total_time_ms FROM card_stats WHERE card_id = 500")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(row.0, 500);
    assert_eq!(row.1, 3);
    assert_eq!(row.2, 29000); // 8000 + 15000 + 6000
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sync_anki_collection_convenience_delegates() {
    let (pool, _container) = setup_pg().await;
    let db = create_test_anki_db();

    let stats = sync_anki_collection(&pool, db.path()).await.unwrap();

    assert_eq!(stats.decks_upserted, 2);
    assert_eq!(stats.notes_upserted, 2);
    assert_eq!(stats.cards_upserted, 2);
}

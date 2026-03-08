use std::collections::HashMap;

use database::connection::{with_connection, with_transaction};
use database::migrations::{MigrationResult, run_migrations};
use database::pool::{check_connection, create_pool};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

/// Spin up a Postgres container and return a connected pool.
async fn setup_pool() -> (PgPool, testcontainers::ContainerAsync<Postgres>) {
    let container = Postgres::default().start().await.unwrap();
    let host = container.get_host().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@{host}:{port}/postgres");

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&url)
        .await
        .unwrap();

    (pool, container)
}

/// Build a Settings instance pointing at the testcontainer.
fn settings_for_container(host: &str, port: u16) -> common::config::Settings {
    common::config::Settings {
        postgres_url: format!("postgresql://postgres:postgres@{host}:{port}/postgres"),
        qdrant_url: "http://localhost:6333".to_string(),
        qdrant_quantization: common::config::Quantization::None,
        qdrant_on_disk: false,
        redis_url: "redis://localhost:6379/0".to_string(),
        job_queue_name: "test_jobs".to_string(),
        job_result_ttl_seconds: 3600,
        job_max_retries: 3,
        embedding_provider: "mock".to_string(),
        embedding_model: "test".to_string(),
        embedding_dimension: 384,
        rerank_enabled: false,
        rerank_model: "test".to_string(),
        rerank_top_n: 10,
        rerank_batch_size: 32,
        api_host: "0.0.0.0".to_string(),
        api_port: 8000,
        api_key: None,
        debug: false,
        anki_collection_path: None,
    }
}

// ============================================================
// Pool tests
// ============================================================

#[tokio::test]
async fn test_create_pool_connects_successfully() {
    let container = Postgres::default().start().await.unwrap();
    let host = container.get_host().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let settings = settings_for_container(&host.to_string(), port);

    let pool = create_pool(&settings)
        .await
        .expect("create_pool should succeed");

    // Pool should be usable
    let row: (i32,) = sqlx::query_as("SELECT 1").fetch_one(&pool).await.unwrap();
    assert_eq!(row.0, 1);
}

#[tokio::test]
async fn test_create_pool_fails_with_bad_url() {
    let mut settings = settings_for_container("localhost", 1);
    settings.postgres_url = "postgresql://localhost:1/nonexistent".to_string();

    let result = create_pool(&settings).await;
    assert!(
        result.is_err(),
        "create_pool should fail with unreachable database"
    );
}

#[tokio::test]
async fn test_check_connection_returns_true_for_live_db() {
    let (pool, _container) = setup_pool().await;
    assert!(
        check_connection(&pool).await,
        "check_connection should return true for live db"
    );
}

#[tokio::test]
async fn test_check_connection_returns_false_for_closed_pool() {
    let (pool, container) = setup_pool().await;
    // Drop the container to make the database unreachable
    drop(container);
    // Give it a moment for the connection to become invalid
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    assert!(
        !check_connection(&pool).await,
        "check_connection should return false when db is unreachable"
    );
}

// ============================================================
// Migration tests
// ============================================================

#[tokio::test]
async fn test_run_migrations_creates_schema_migrations_table() {
    let (pool, _container) = setup_pool().await;

    let _result = run_migrations(&pool)
        .await
        .expect("migrations should succeed");

    // Verify schema_migrations table exists
    let exists: bool = sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'schema_migrations')",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert!(
        exists,
        "schema_migrations table should exist after migrations"
    );
}

#[tokio::test]
async fn test_run_migrations_applies_initial_schema() {
    let (pool, _container) = setup_pool().await;

    let result = run_migrations(&pool)
        .await
        .expect("migrations should succeed");

    assert!(
        result.applied.contains(&"001_initial_schema".to_string()),
        "001_initial_schema should be in applied list"
    );

    // Verify key tables were created
    let tables: Vec<String> = sqlx::query_scalar(
        "SELECT table_name::text FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name",
    )
    .fetch_all(&pool)
    .await
    .unwrap();

    let expected_tables = [
        "card_stats",
        "cards",
        "decks",
        "models",
        "note_topics",
        "notes",
        "schema_migrations",
        "sync_metadata",
        "topics",
    ];
    for table in &expected_tables {
        assert!(
            tables.contains(&table.to_string()),
            "table '{table}' should exist after 001_initial_schema"
        );
    }
}

#[tokio::test]
async fn test_run_migrations_applies_trigram_search() {
    let (pool, _container) = setup_pool().await;

    let result = run_migrations(&pool)
        .await
        .expect("migrations should succeed");

    assert!(
        result
            .applied
            .contains(&"002_pg_trgm_lexical_search".to_string()),
        "002_pg_trgm_lexical_search should be in applied list"
    );
}

#[tokio::test]
async fn test_run_migrations_is_idempotent() {
    let (pool, _container) = setup_pool().await;

    // First run: both should be applied
    let first = run_migrations(&pool)
        .await
        .expect("first run should succeed");
    assert_eq!(
        first.applied.len(),
        2,
        "first run should apply 2 migrations"
    );
    assert_eq!(first.skipped.len(), 0, "first run should skip nothing");

    // Second run: both should be skipped
    let second = run_migrations(&pool)
        .await
        .expect("second run should succeed");
    assert_eq!(second.applied.len(), 0, "second run should apply nothing");
    assert_eq!(
        second.skipped.len(),
        2,
        "second run should skip 2 migrations"
    );
}

#[tokio::test]
async fn test_run_migrations_returns_correct_applied_and_skipped() {
    let (pool, _container) = setup_pool().await;

    let result = run_migrations(&pool)
        .await
        .expect("migrations should succeed");

    // Both migrations should be applied on first run
    assert_eq!(
        result.applied,
        vec!["001_initial_schema", "002_pg_trgm_lexical_search"]
    );
    assert!(result.skipped.is_empty());
}

// ============================================================
// Connection helper tests
// ============================================================

#[tokio::test]
async fn test_with_connection_executes_closure() {
    let (pool, _container) = setup_pool().await;

    let value = with_connection(&pool, |conn| {
        Box::pin(async move {
            let row: (i32,) = sqlx::query_as("SELECT 42")
                .fetch_one(&mut *conn)
                .await
                .map_err(|e| common::error::AnkiAtlasError::DatabaseConnection {
                    message: e.to_string(),
                    context: HashMap::new(),
                })?;
            Ok(row.0)
        })
    })
    .await
    .expect("with_connection should succeed");

    assert_eq!(value, 42);
}

#[tokio::test]
async fn test_with_transaction_commits_on_success() {
    let (pool, _container) = setup_pool().await;

    // Create a temp table
    sqlx::query("CREATE TABLE test_txn (id INT PRIMARY KEY, val TEXT)")
        .execute(&pool)
        .await
        .unwrap();

    // Insert via transaction
    with_transaction(&pool, |txn| {
        Box::pin(async move {
            sqlx::query("INSERT INTO test_txn (id, val) VALUES (1, 'committed')")
                .execute(&mut **txn)
                .await
                .map_err(|e| common::error::AnkiAtlasError::DatabaseConnection {
                    message: e.to_string(),
                    context: HashMap::new(),
                })?;
            Ok(())
        })
    })
    .await
    .expect("with_transaction should succeed");

    // Verify data was committed
    let val: String = sqlx::query_scalar("SELECT val FROM test_txn WHERE id = 1")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(val, "committed");
}

#[tokio::test]
async fn test_with_transaction_rolls_back_on_error() {
    let (pool, _container) = setup_pool().await;

    // Create a temp table
    sqlx::query("CREATE TABLE test_rollback (id INT PRIMARY KEY, val TEXT)")
        .execute(&pool)
        .await
        .unwrap();

    // Insert then return error -- should rollback
    let result: common::error::Result<()> = with_transaction(&pool, |txn| {
        Box::pin(async move {
            sqlx::query("INSERT INTO test_rollback (id, val) VALUES (1, 'should_rollback')")
                .execute(&mut **txn)
                .await
                .map_err(|e| common::error::AnkiAtlasError::DatabaseConnection {
                    message: e.to_string(),
                    context: HashMap::new(),
                })?;
            // Return an error to trigger rollback
            Err(common::error::AnkiAtlasError::DatabaseConnection {
                message: "intentional error".to_string(),
                context: HashMap::new(),
            })
        })
    })
    .await;

    assert!(
        result.is_err(),
        "with_transaction should propagate the error"
    );

    // Verify data was NOT committed
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM test_rollback")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(count, 0, "rolled back transaction should leave table empty");
}

// ============================================================
// Send + Sync compile-time checks
// ============================================================

#[test]
fn migration_result_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MigrationResult>();
}

// ============================================================
// Embedded SQL file identity check
// ============================================================

#[test]
fn embedded_migrations_match_python_source() {
    // The migration SQL files in crates/database/migrations/ should be
    // byte-identical to packages/common/migrations/. We verify by checking
    // the embedded content contains expected markers.
    let migrations = database::migrations::MIGRATIONS;
    assert_eq!(migrations.len(), 2);
    assert_eq!(migrations[0].0, "001_initial_schema");
    assert_eq!(migrations[1].0, "002_pg_trgm_lexical_search");

    // Verify key content from migration 1
    assert!(
        migrations[0].1.contains("CREATE TABLE IF NOT EXISTS decks"),
        "001 should contain decks table"
    );
    assert!(
        migrations[0].1.contains("CREATE TABLE IF NOT EXISTS notes"),
        "001 should contain notes table"
    );

    // Verify key content from migration 2
    assert!(
        migrations[1]
            .1
            .contains("CREATE EXTENSION IF NOT EXISTS pg_trgm"),
        "002 should enable pg_trgm extension"
    );
}

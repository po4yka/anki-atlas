# Testing Patterns Reference

Real examples from this codebase. Use these as templates when writing tests.

---

## mockall: automock Trait

Use `#[cfg_attr(test, mockall::automock)]` on traits for automatic mock generation.
Works when tests are in the **same crate** as the trait.

**Source:** `crates/llm/src/provider.rs:26`

```rust
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait LlmProvider: Send + Sync {
    async fn generate(
        &self,
        model: &str,
        prompt: &str,
        opts: &GenerateOptions,
    ) -> Result<LlmResponse, LlmError>;

    async fn check_connection(&self) -> bool;

    async fn list_models(&self) -> Result<Vec<String>, LlmError>;
}
```

Usage in tests:

```rust
let mut mock = MockLlmProvider::new();
mock.expect_generate()
    .with(eq("model"), eq("prompt"), always())
    .returning(|_, _, _| Ok(LlmResponse { /* ... */ }));
```

---

## mockall: Manual mock! Block

Use `mock! {}` when mocking traits from **other crates** (cross-crate boundary).
Required for integration tests in `bins/` that mock traits from `crates/`.

**Source:** `bins/api/tests/handlers_test.rs:29`

```rust
mock! {
    pub Jobs {}

    #[async_trait]
    impl JobManager for Jobs {
        async fn enqueue_sync_job(
            &self,
            payload: SyncJobPayload,
            run_at: Option<chrono::DateTime<chrono::Utc>>,
        ) -> Result<JobRecord, JobError>;

        async fn enqueue_index_job(
            &self,
            payload: IndexJobPayload,
            run_at: Option<chrono::DateTime<chrono::Utc>>,
        ) -> Result<JobRecord, JobError>;

        async fn get_job(&self, job_id: &str) -> Result<JobRecord, JobError>;
        async fn cancel_job(&self, job_id: &str) -> Result<JobRecord, JobError>;
        async fn close(&self) -> Result<(), JobError>;
    }
}
```

Usage: `MockJobs::new()` -- the generated type is `Mock` + the block name.

**Decision tree:**
- Test in same crate as trait? -> `#[cfg_attr(test, mockall::automock)]`
- Test in different crate (e.g., `bins/` testing `crates/` traits)? -> `mock! {}`

---

## testcontainers: PostgreSQL

Use for integration tests that need a real database. Requires Docker.
Skip gracefully when Docker is unavailable.

**Source:** `crates/database/tests/database_tests.rs:11`

```rust
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

async fn setup_pool() -> Option<(PgPool, testcontainers::ContainerAsync<Postgres>)> {
    let container = match Postgres::default().start().await {
        Ok(container) => container,
        Err(error) => {
            eprintln!("skipping postgres-backed database test: {error}");
            return None;
        }
    };
    let host = container.get_host().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@{host}:{port}/postgres");

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&url)
        .await
        .unwrap();

    Some((pool, container))
}
```

**Important:** Hold the container in scope for the test duration. Dropping it stops the container.

---

## wiremock: HTTP Mock Server

Use for testing HTTP clients without hitting real APIs.

**Source:** `crates/anki-reader/tests/test_connect.rs`

```rust
use wiremock::matchers::{body_json, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn version_sends_correct_payload() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(body_json(json!({
            "action": "version",
            "version": 6
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "result": 6,
            "error": null
        })))
        .mount(&server)
        .await;

    let client = AnkiConnectClient::new(&server.uri(), 5);
    let result = client.version().await.unwrap();
    assert_eq!(result, 6);
}
```

For partial body matching: use `body_partial_json` from `wiremock::matchers`.

---

## tempfile: Ephemeral Files

Use `NamedTempFile` for tests that need real filesystem paths. Auto-cleaned on drop.

**Source:** `crates/anki-sync/tests/test_core.rs:12`

```rust
use rusqlite::Connection;
use tempfile::NamedTempFile;

fn create_test_anki_db() -> NamedTempFile {
    let file = NamedTempFile::new().expect("create temp file");
    let conn = Connection::open(file.path()).expect("open sqlite");

    conn.execute_batch("
        CREATE TABLE col (
            id INTEGER PRIMARY KEY,
            -- ... schema ...
        );
        INSERT INTO col VALUES (/* ... */);
    ")
    .expect("create test db");

    file
}
```

For directories: use `tempfile::TempDir` instead.

---

## Test Helpers / Builders

Common patterns for reducing test boilerplate:

### Setup function returning Option (graceful skip)

```rust
// When external dependency (Docker) may be unavailable
async fn setup() -> Option<TestContext> { /* ... */ }

#[tokio::test]
async fn test_something() {
    let Some(ctx) = setup().await else { return };
    // ... test body ...
}
```

### Builder pattern for test data

```rust
fn make_note(front: &str, back: &str) -> Note {
    Note {
        id: NoteId(1),
        model_id: 1234567890,
        fields: vec![front.to_string(), back.to_string()],
        tags: vec![],
        // ... sensible defaults ...
    }
}
```

### Naming convention

```
test_<behavior>_<condition>_<expected>
```

Examples from the codebase:
- `slugify_simple_text`
- `slugify_empty_string`
- `normalize_tag_unknown_kebab`
- `version_sends_correct_payload`
- `find_notes_sends_params`

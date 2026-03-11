#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

use anki_atlas_api::router::build_router;
use anki_atlas_api::schemas::SearchRequest;
use anki_atlas_api::services::{build_api_services, build_app_state};
use anyhow::{Context, Result, anyhow, bail};
use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Method, Request, StatusCode, header::CONTENT_TYPE},
};
use common::config::{EmbeddingProviderKind, Quantization, Settings};
use rusqlite::Connection;
use serde_json::Value;
use sqlx::PgPool;
use tempfile::TempDir;
use testcontainers::{
    ContainerAsync, GenericImage,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};
use testcontainers_modules::postgres::Postgres;
use tower::ServiceExt;
use uuid::Uuid;

const QDRANT_COLLECTION_NAME: &str = "anki_notes";
const QDRANT_IMAGE_TAG: &str = "v1.16.3";

pub struct SeedNote<'a> {
    pub note_id: i64,
    pub card_id: i64,
    pub deck_id: i64,
    pub deck_name: &'a str,
    pub model_id: i64,
    pub model_name: &'a str,
    pub tags: Vec<&'a str>,
    pub normalized_text: &'a str,
    pub raw_fields: &'a str,
    pub ivl: i32,
    pub lapses: i32,
    pub reps: i32,
    pub fail_rate: Option<f32>,
}

pub struct WorkerProcess {
    child: Child,
}

impl WorkerProcess {
    pub fn stop(&mut self) -> Result<()> {
        if let Some(_status) = self.child.try_wait()? {
            return Ok(());
        }

        self.child.kill()?;
        let _ = self.child.wait()?;
        Ok(())
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

pub struct TestStack {
    _postgres: ContainerAsync<Postgres>,
    _redis: ContainerAsync<GenericImage>,
    _qdrant: ContainerAsync<GenericImage>,
    pub fixture_dir: TempDir,
    pub pool: PgPool,
    pub app: Router,
    pub qdrant_url: String,
    pub env: HashMap<String, String>,
}

impl TestStack {
    pub async fn new(api_key: Option<&str>) -> Result<Self> {
        let postgres = Postgres::default()
            .start()
            .await
            .context("start postgres container")?;
        let redis = start_redis_container().await?;
        let qdrant = GenericImage::new("qdrant/qdrant", QDRANT_IMAGE_TAG)
            .with_exposed_port(6333.tcp())
            .with_exposed_port(6334.tcp())
            .start()
            .await
            .context("start qdrant container")?;

        let postgres_host = postgres.get_host().await.context("postgres host")?;
        let postgres_port = postgres
            .get_host_port_ipv4(5432)
            .await
            .context("postgres port")?;
        let redis_host = redis.get_host().await.context("redis host")?;
        let redis_port = redis.get_host_port_ipv4(6379).await.context("redis port")?;
        let qdrant_host = qdrant.get_host().await.context("qdrant host")?;
        let qdrant_port = qdrant
            .get_host_port_ipv4(6333)
            .await
            .context("qdrant port")?;
        let qdrant_grpc_port = qdrant
            .get_host_port_ipv4(6334)
            .await
            .context("qdrant grpc port")?;

        let queue_name = format!("test:jobs:{}", Uuid::new_v4());
        let qdrant_url = format!("http://{qdrant_host}:{qdrant_port}");
        let settings = build_settings(
            format!("postgresql://postgres:postgres@{postgres_host}:{postgres_port}/postgres"),
            format!("http://{qdrant_host}:{qdrant_grpc_port}"),
            format!("redis://{redis_host}:{redis_port}/0"),
            queue_name.clone(),
            api_key.map(ToOwned::to_owned),
        );

        wait_for_qdrant_ready(&qdrant_url).await?;

        let pool = database::create_pool(&settings.database())
            .await
            .context("create test postgres pool")?;
        database::run_migrations(&pool)
            .await
            .context("run test migrations")?;

        let services = build_api_services(&settings)
            .await
            .context("build real api services")?;
        let app = build_router(build_app_state(settings.api(), services));
        let fixture_dir = TempDir::new().context("create fixture tempdir")?;
        let mut env = default_env(&settings);
        env.insert(
            "ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER".to_string(),
            "1".to_string(),
        );

        let stack = Self {
            _postgres: postgres,
            _redis: redis,
            _qdrant: qdrant,
            fixture_dir,
            pool,
            app,
            qdrant_url,
            env,
        };

        stack.reset_state().await?;
        Ok(stack)
    }

    pub async fn reset_state(&self) -> Result<()> {
        self.truncate_app_tables().await?;
        self.reset_qdrant_collection().await?;
        Ok(())
    }

    pub async fn truncate_app_tables(&self) -> Result<()> {
        sqlx::query(
            "TRUNCATE TABLE
                card_stats,
                cards,
                note_topics,
                notes,
                decks,
                models,
                topics,
                sync_metadata
             RESTART IDENTITY CASCADE",
        )
        .execute(&self.pool)
        .await
        .context("truncate app tables")?;
        Ok(())
    }

    pub async fn reset_qdrant_collection(&self) -> Result<()> {
        let client = reqwest::Client::new();
        let response = client
            .delete(format!(
                "{}/collections/{QDRANT_COLLECTION_NAME}",
                self.qdrant_url
            ))
            .send()
            .await
            .context("delete qdrant collection")?;

        if !response.status().is_success() && response.status() != StatusCode::NOT_FOUND {
            bail!("unexpected qdrant delete status: {}", response.status());
        }

        Ok(())
    }

    pub fn create_anki_fixture(&self, file_name: &str) -> Result<PathBuf> {
        let path = self.fixture_dir.path().join(file_name);
        let conn = Connection::open(&path).context("open sqlite fixture")?;

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
            INSERT INTO notes VALUES (100, 'fixture-guid', 1234567890, 1700000000, -1, ' rust ownership ', 'Ownership front\x1fOwnership back', 'Ownership front', 0, 0, '');

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
        .context("seed sqlite fixture")?;

        Ok(path)
    }

    pub fn create_taxonomy_fixture(&self, file_name: &str) -> Result<PathBuf> {
        let path = self.fixture_dir.path().join(file_name);
        fs::write(
            &path,
            r#"topics:
  - path: rust
    label: Rust
    description: Systems programming and memory safety.
    children:
      - path: rust/ownership
        label: Ownership
        description: Ownership rules and move semantics.
      - path: rust/borrowing
        label: Borrowing
        description: Shared and mutable borrowing.
"#,
        )
        .context("write taxonomy fixture")?;
        Ok(path)
    }

    pub async fn seed_note(&self, seed: SeedNote<'_>) -> Result<()> {
        let tags = seed
            .tags
            .into_iter()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();

        sqlx::query(
            "INSERT INTO decks (deck_id, name, parent_name) \
             VALUES ($1, $2, NULL) \
             ON CONFLICT (deck_id) DO UPDATE SET name = EXCLUDED.name",
        )
        .bind(seed.deck_id)
        .bind(seed.deck_name)
        .execute(&self.pool)
        .await
        .context("upsert deck")?;

        sqlx::query(
            "INSERT INTO models (model_id, name, fields, templates) \
             VALUES ($1, $2, $3, $4) \
             ON CONFLICT (model_id) DO UPDATE SET name = EXCLUDED.name",
        )
        .bind(seed.model_id)
        .bind(seed.model_name)
        .bind(serde_json::json!(["Front", "Back"]))
        .bind(serde_json::json!(["Card 1"]))
        .execute(&self.pool)
        .await
        .context("upsert model")?;

        sqlx::query(
            "INSERT INTO notes \
             (note_id, model_id, tags, fields_json, raw_fields, normalized_text, mtime, usn) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        )
        .bind(seed.note_id)
        .bind(seed.model_id)
        .bind(&tags)
        .bind(serde_json::json!({
            "Front": seed.normalized_text,
            "Back": format!("Answer for {}", seed.note_id),
        }))
        .bind(seed.raw_fields)
        .bind(seed.normalized_text)
        .bind(seed.note_id)
        .bind(0_i32)
        .execute(&self.pool)
        .await
        .with_context(|| format!("insert note {}", seed.note_id))?;

        sqlx::query(
            "INSERT INTO cards \
             (card_id, note_id, deck_id, ord, due, ivl, ease, lapses, reps, queue, type, mtime, usn) \
             VALUES ($1, $2, $3, 0, 0, $4, 2500, $5, $6, 0, 2, $7, 0)",
        )
        .bind(seed.card_id)
        .bind(seed.note_id)
        .bind(seed.deck_id)
        .bind(seed.ivl)
        .bind(seed.lapses)
        .bind(seed.reps)
        .bind(seed.note_id)
        .execute(&self.pool)
        .await
        .with_context(|| format!("insert card {}", seed.card_id))?;

        sqlx::query(
            "INSERT INTO card_stats \
             (card_id, reviews, avg_ease, fail_rate, last_review_at, total_time_ms) \
             VALUES ($1, $2, 2.5, $3, NOW(), $4)",
        )
        .bind(seed.card_id)
        .bind(seed.reps)
        .bind(seed.fail_rate)
        .bind(i64::from(seed.reps) * 1_000_i64)
        .execute(&self.pool)
        .await
        .with_context(|| format!("insert card stats {}", seed.card_id))?;

        Ok(())
    }

    pub async fn assign_topic(
        &self,
        note_id: i64,
        topic_path: &str,
        confidence: f32,
        method: &str,
    ) -> Result<()> {
        let topic_id = self.topic_id(topic_path).await?;
        sqlx::query(
            "INSERT INTO note_topics (note_id, topic_id, confidence, method) \
             VALUES ($1, $2, $3, $4) \
             ON CONFLICT (note_id, topic_id) DO UPDATE \
             SET confidence = EXCLUDED.confidence, method = EXCLUDED.method",
        )
        .bind(note_id)
        .bind(topic_id as i32)
        .bind(confidence)
        .bind(method)
        .execute(&self.pool)
        .await
        .with_context(|| format!("assign topic {topic_path} to note {note_id}"))?;
        Ok(())
    }

    pub async fn topic_id(&self, topic_path: &str) -> Result<i64> {
        let topic_id = sqlx::query_scalar::<_, i32>("SELECT topic_id FROM topics WHERE path = $1")
            .bind(topic_path)
            .fetch_one(&self.pool)
            .await
            .with_context(|| format!("fetch topic id for {topic_path}"))?;
        Ok(i64::from(topic_id))
    }

    pub fn run_cli(&self, args: &[&str]) -> Result<Output> {
        let cli = build_binary("anki-atlas")?;
        Command::new(cli)
            .current_dir(workspace_root()?)
            .envs(&self.env)
            .args(args)
            .output()
            .context("run anki-atlas cli")
    }

    pub fn spawn_worker(&self) -> Result<WorkerProcess> {
        let worker = build_binary("anki-atlas-worker")?;
        let child = Command::new(worker)
            .current_dir(workspace_root()?)
            .envs(&self.env)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .context("spawn anki-atlas-worker")?;

        Ok(WorkerProcess { child })
    }

    pub async fn notes_count(&self) -> Result<i64> {
        let count = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM notes")
            .fetch_one(&self.pool)
            .await
            .context("count notes")?;
        Ok(count)
    }

    pub async fn cards_count(&self) -> Result<i64> {
        let count = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM cards")
            .fetch_one(&self.pool)
            .await
            .context("count cards")?;
        Ok(count)
    }

    pub async fn qdrant_point_count(&self) -> Result<usize> {
        let client = reqwest::Client::new();
        let response = client
            .get(format!(
                "{}/collections/{QDRANT_COLLECTION_NAME}",
                self.qdrant_url
            ))
            .send()
            .await
            .context("fetch qdrant collection info")?;

        if response.status() == StatusCode::NOT_FOUND {
            return Ok(0);
        }
        if !response.status().is_success() {
            bail!("unexpected qdrant collection status: {}", response.status());
        }

        let body: Value = response.json().await.context("decode qdrant response")?;
        Ok(body["result"]["points_count"].as_u64().unwrap_or(0) as usize)
    }

    pub async fn poll_job_terminal(&self, job_id: &str, timeout: Duration) -> Result<Value> {
        let deadline = Instant::now() + timeout;

        loop {
            let (status_code, body) = self.get_json(&format!("/jobs/{job_id}"), &[]).await?;
            if status_code != StatusCode::OK {
                bail!("unexpected job status response {status_code}: {body}");
            }
            let status = body["status"].as_str().unwrap_or_default();

            if matches!(status, "succeeded" | "failed" | "cancelled") {
                return Ok(body);
            }

            if Instant::now() >= deadline {
                bail!("job {job_id} did not reach a terminal state in {timeout:?}");
            }

            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    pub async fn api_search(&self, query: &str) -> Result<Value> {
        let (status_code, body) = self
            .post_json(
                "/search",
                &SearchRequest {
                    query: query.to_string(),
                    filters: None,
                    limit: 10,
                    semantic_weight: 1.0,
                    fts_weight: 1.0,
                    semantic_only: false,
                    fts_only: false,
                    rerank_override: None,
                    rerank_top_n_override: None,
                },
                &[],
            )
            .await?;
        if status_code != StatusCode::OK {
            bail!("unexpected search response {status_code}: {body}");
        }
        Ok(body)
    }

    pub async fn get_json(
        &self,
        path: &str,
        headers: &[(&str, &str)],
    ) -> Result<(StatusCode, Value)> {
        self.send_json_request::<()>(Method::GET, path, None, headers)
            .await
    }

    pub async fn post_json<T: serde::Serialize>(
        &self,
        path: &str,
        body: &T,
        headers: &[(&str, &str)],
    ) -> Result<(StatusCode, Value)> {
        self.send_json_request(Method::POST, path, Some(body), headers)
            .await
    }

    async fn send_json_request<T: serde::Serialize>(
        &self,
        method: Method,
        path: &str,
        body: Option<&T>,
        headers: &[(&str, &str)],
    ) -> Result<(StatusCode, Value)> {
        let mut builder = Request::builder().method(method).uri(path);
        for (name, value) in headers {
            builder = builder.header(*name, *value);
        }

        let request = if let Some(body) = body {
            let payload = serde_json::to_vec(body).context("encode request body")?;
            builder
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(payload))
                .context("build json request")?
        } else {
            builder.body(Body::empty()).context("build empty request")?
        };

        let response = self
            .app
            .clone()
            .oneshot(request)
            .await
            .context("dispatch in-process request")?;
        let status_code = response.status();
        let bytes = to_bytes(response.into_body(), 1024 * 1024)
            .await
            .context("read response body")?;
        let json = if bytes.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&bytes).context("decode response body")?
        };

        Ok((status_code, json))
    }
}

fn build_settings(
    postgres_url: String,
    qdrant_url: String,
    redis_url: String,
    queue_name: String,
    api_key: Option<String>,
) -> Settings {
    Settings {
        postgres_url,
        qdrant_url,
        qdrant_quantization: Quantization::None,
        qdrant_on_disk: false,
        redis_url,
        job_queue_name: queue_name,
        job_result_ttl_seconds: 300,
        job_max_retries: 3,
        embedding_provider: EmbeddingProviderKind::Mock,
        embedding_model: "mock/test".to_string(),
        embedding_dimension: 384,
        rerank_enabled: false,
        rerank_model: "test-rerank".to_string(),
        rerank_top_n: 10,
        rerank_batch_size: 16,
        api_host: "127.0.0.1".to_string(),
        api_port: 0,
        api_key,
        debug: false,
        anki_collection_path: None,
    }
}

fn default_env(settings: &Settings) -> HashMap<String, String> {
    HashMap::from([
        (
            "ANKIATLAS_POSTGRES_URL".to_string(),
            settings.postgres_url.clone(),
        ),
        (
            "ANKIATLAS_QDRANT_URL".to_string(),
            settings.qdrant_url.clone(),
        ),
        (
            "ANKIATLAS_REDIS_URL".to_string(),
            settings.redis_url.clone(),
        ),
        (
            "ANKIATLAS_JOB_QUEUE_NAME".to_string(),
            settings.job_queue_name.clone(),
        ),
        (
            "ANKIATLAS_JOB_RESULT_TTL_SECONDS".to_string(),
            settings.job_result_ttl_seconds.to_string(),
        ),
        (
            "ANKIATLAS_JOB_MAX_RETRIES".to_string(),
            settings.job_max_retries.to_string(),
        ),
        (
            "ANKIATLAS_EMBEDDING_PROVIDER".to_string(),
            "mock".to_string(),
        ),
        (
            "ANKIATLAS_EMBEDDING_MODEL".to_string(),
            settings.embedding_model.clone(),
        ),
        (
            "ANKIATLAS_EMBEDDING_DIMENSION".to_string(),
            settings.embedding_dimension.to_string(),
        ),
        ("ANKIATLAS_RERANK_ENABLED".to_string(), "false".to_string()),
        ("ANKIATLAS_API_HOST".to_string(), settings.api_host.clone()),
        (
            "ANKIATLAS_API_PORT".to_string(),
            settings.api_port.to_string(),
        ),
        (
            "ANKIATLAS_QDRANT_QUANTIZATION".to_string(),
            "none".to_string(),
        ),
        ("ANKIATLAS_QDRANT_ON_DISK".to_string(), "false".to_string()),
        (
            "ANKIATLAS_API_KEY".to_string(),
            settings.api_key.clone().unwrap_or_default(),
        ),
    ])
}

async fn start_redis_container() -> Result<ContainerAsync<GenericImage>> {
    let mut last_error = None;

    for attempt in 1..=3 {
        match GenericImage::new("redis", "7-alpine")
            .with_exposed_port(6379.tcp())
            .with_wait_for(WaitFor::message_on_stdout("Ready to accept connections"))
            .start()
            .await
        {
            Ok(container) => return Ok(container),
            Err(error) => {
                last_error = Some(error);
                if attempt < 3 {
                    tokio::time::sleep(Duration::from_secs(attempt)).await;
                }
            }
        }
    }

    Err(anyhow!(
        "start redis container after retries: {}",
        last_error.expect("redis startup should produce an error")
    ))
}

async fn wait_for_qdrant_ready(qdrant_url: &str) -> Result<()> {
    let client = reqwest::Client::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    let readiness_paths = ["readyz", "healthz"];

    loop {
        for path in readiness_paths {
            let response = client.get(format!("{qdrant_url}/{path}")).send().await;
            if let Ok(response) = response
                && response.status().is_success()
            {
                return Ok(());
            }
        }

        if Instant::now() >= deadline {
            bail!("qdrant did not become ready in time");
        }

        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn build_binary(name: &str) -> Result<PathBuf> {
    let workspace_manifest = workspace_root()?.join("Cargo.toml");
    let artifact = escargot::CargoBuild::new()
        .manifest_path(&workspace_manifest)
        .bin(name)
        .run()
        .with_context(|| format!("build binary {name} with escargot"))?;
    Ok(artifact.path().to_path_buf())
}

fn workspace_root() -> Result<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .context("resolve workspace root")?;
    if !root.join("Cargo.toml").exists() {
        return Err(anyhow!("workspace root cargo manifest not found"));
    }
    Ok(root)
}

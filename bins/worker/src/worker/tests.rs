use super::*;
use crate::config::WorkerConfig;
use crate::envelope::JobEnvelope;
use jobs::{
    IndexJobPayload, JobManager, JobPayload, JobRecord, JobStatus, JobType, PgJobManager,
    SyncJobPayload,
};
use std::future::Future;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use testcontainers::{
    GenericImage,
    core::{IntoContainerPort, WaitFor},
    runners::AsyncRunner,
};

fn test_config() -> WorkerConfig {
    WorkerConfig {
        postgres_url: "postgres://localhost:5432/anki_atlas_test".to_string(),
        queue_name: "test:queue".to_string(),
        max_concurrency: 2,
        max_retries: 3,
        poll_interval: Duration::from_millis(50),
        allow_abort_on_shutdown: true,
        result_ttl_seconds: 3600,
    }
}

fn test_job_record(job_id: &str, status: JobStatus) -> JobRecord {
    JobRecord {
        job_id: job_id.to_string(),
        job_type: JobType::Sync,
        status,
        payload: JobPayload::Sync(SyncJobPayload {
            source: "/tmp/collection.anki2".to_string(),
            run_migrations: true,
            index: true,
            reindex_mode: common::ReindexMode::Incremental,
        }),
        progress: 0.0,
        message: None,
        attempts: 0,
        max_retries: 3,
        cancel_requested: false,
        created_at: None,
        scheduled_for: None,
        started_at: None,
        finished_at: None,
        result: None,
        error: None,
    }
}

fn test_envelope(job_id: &str, job_type: JobType) -> JobEnvelope {
    let payload = match job_type {
        JobType::Sync => JobPayload::Sync(SyncJobPayload {
            source: "/tmp/collection.anki2".to_string(),
            run_migrations: true,
            index: true,
            reindex_mode: common::ReindexMode::Incremental,
        }),
        JobType::Index => JobPayload::Index(IndexJobPayload {
            reindex_mode: common::ReindexMode::Incremental,
        }),
    };

    JobEnvelope {
        job_id: job_id.to_string(),
        job_type,
        payload,
    }
}

async fn wait_until<F>(timeout: Duration, mut condition: F)
where
    F: FnMut() -> bool,
{
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if condition() {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "condition not met within {:?}",
            timeout
        );
        tokio::task::yield_now().await;
    }
}

async fn wait_until_async<F, Fut>(timeout: Duration, mut condition: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if condition().await {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "async condition not met within {:?}",
            timeout
        );
        tokio::task::yield_now().await;
    }
}

async fn start_postgres_container() -> anyhow::Result<testcontainers::ContainerAsync<GenericImage>>
{
    let mut last_error = None;

    for attempt in 1..=3 {
        match GenericImage::new("postgres", "16-alpine")
            .with_exposed_port(5432.tcp())
            .with_env_var("POSTGRES_PASSWORD", "postgres")
            .with_wait_for(WaitFor::message_on_stderr(
                "database system is ready to accept connections",
            ))
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

    Err(anyhow::anyhow!(
        "start postgres container after retries: {}",
        last_error.expect("postgres startup should produce an error")
    ))
}

async fn shutdown_and_join<Q: QueueBackend + 'static>(
    worker: &Arc<Worker<Q>>,
    handle: tokio::task::JoinHandle<anyhow::Result<()>>,
) {
    worker.shutdown(Duration::from_secs(1)).await;
    let result = tokio::time::timeout(Duration::from_secs(2), handle).await;
    assert!(result.is_ok(), "worker task should shut down cleanly");
}

struct RealPgBackend {
    pool: sqlx::PgPool,
}

impl RealPgBackend {
    async fn connect(postgres_url: &str) -> anyhow::Result<Self> {
        let pool = jobs::connection::create_job_pool(postgres_url)
            .await
            .map_err(anyhow::Error::from)?;
        jobs::persistence::ensure_schema(&pool)
            .await
            .map_err(anyhow::Error::from)?;
        Ok(Self { pool })
    }
}

impl QueueBackend for RealPgBackend {
    async fn brpop(&self, _key: &str, timeout: f64) -> anyhow::Result<Option<String>> {
        match jobs::persistence::pop_next_job(&self.pool).await? {
            Some(record) => {
                let envelope = jobs::JobEnvelope::from(&record);
                let json = serde_json::to_string(&envelope)?;
                Ok(Some(json))
            }
            None => {
                tokio::time::sleep(std::time::Duration::from_secs_f64(timeout)).await;
                Ok(None)
            }
        }
    }

    async fn lpush(&self, _key: &str, value: &str) -> anyhow::Result<()> {
        let envelope: jobs::JobEnvelope = serde_json::from_str(value)?;
        let record = self.load_job_record(&envelope.job_id).await?;
        if let Some(record) = record {
            jobs::persistence::reenqueue_job(&self.pool, &record).await?;
        }
        Ok(())
    }

    async fn load_job_record(&self, job_id: &str) -> anyhow::Result<Option<JobRecord>> {
        jobs::persistence::load_job_record(&self.pool, job_id)
            .await
            .map_err(anyhow::Error::from)
    }

    async fn save_job_record(&self, record: &JobRecord, ttl_seconds: u64) -> anyhow::Result<()> {
        jobs::persistence::save_job_record(&self.pool, record, ttl_seconds)
            .await
            .map_err(anyhow::Error::from)
    }
}

// --- Send + Sync compile-time assertions ---

#[test]
fn worker_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Worker<MockQueueBackend>>();
}

#[test]
fn worker_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<Worker<MockQueueBackend>>();
}

// --- Worker::new ---

#[test]
fn new_creates_semaphore_with_max_concurrency_permits() {
    let config = test_config();
    let max_concurrency = config.max_concurrency;
    let mock = MockQueueBackend::new();

    let worker = Worker::new(config, mock);

    assert_eq!(worker.semaphore.available_permits(), max_concurrency);
}

#[test]
fn new_stores_config() {
    let config = test_config();
    let expected_queue = config.queue_name.clone();
    let mock = MockQueueBackend::new();

    let worker = Worker::new(config, mock);

    assert_eq!(worker.config.queue_name, expected_queue);
}

// --- Shutdown behavior ---

#[tokio::test]
async fn run_blocks_until_shutdown_signal() {
    let config = test_config();
    let mut mock = MockQueueBackend::new();
    let brpop_count = Arc::new(AtomicU32::new(0));
    let brpop_count_clone = Arc::clone(&brpop_count);

    mock.expect_brpop().returning(move |_, _| {
        brpop_count_clone.fetch_add(1, Ordering::SeqCst);
        Ok(None)
    });

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let run_handle = tokio::spawn(async move { worker_clone.run().await });

    wait_until(Duration::from_secs(1), || {
        brpop_count.load(Ordering::SeqCst) > 0
    })
    .await;

    assert!(
        !run_handle.is_finished(),
        "run() should block until shutdown"
    );

    shutdown_and_join(&worker, run_handle).await;
}

#[tokio::test]
async fn shutdown_sets_shutdown_flag() {
    let config = test_config();
    let mock = MockQueueBackend::new();
    let worker = Worker::new(config, mock);

    assert!(!*worker.shutdown_rx.borrow());

    worker.shutdown(Duration::from_secs(1)).await;

    assert!(*worker.shutdown_rx.borrow());
}

// --- Job processing ---

#[tokio::test]
async fn processes_valid_job_envelope_from_queue() {
    let config = test_config();
    let mut mock = MockQueueBackend::new();

    let envelope = test_envelope("job-1", JobType::Sync);
    let envelope_json = serde_json::to_string(&envelope).unwrap();

    let call_count = Arc::new(AtomicU32::new(0));
    let call_count_clone = Arc::clone(&call_count);
    let json_clone = envelope_json.clone();
    mock.expect_brpop().returning(move |_, _| {
        let n = call_count_clone.fetch_add(1, Ordering::SeqCst);
        let json = json_clone.clone();
        if n == 0 { Ok(Some(json)) } else { Ok(None) }
    });

    mock.expect_load_job_record().returning(|id| {
        let record = test_job_record(id, JobStatus::Queued);
        Ok(Some(record))
    });

    let save_count = Arc::new(AtomicU32::new(0));
    let save_count_clone = Arc::clone(&save_count);
    mock.expect_save_job_record().returning(move |_, _| {
        save_count_clone.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move { worker_clone.run().await });

    wait_until(Duration::from_secs(1), || {
        save_count.load(Ordering::SeqCst) > 0
    })
    .await;
    shutdown_and_join(&worker, handle).await;
}

#[tokio::test]
async fn sets_job_status_to_running_before_dispatch() {
    let config = test_config();
    let mut mock = MockQueueBackend::new();

    let envelope = test_envelope("job-2", JobType::Sync);
    let envelope_json = serde_json::to_string(&envelope).unwrap();

    let brpop_count = Arc::new(AtomicU32::new(0));
    let brpop_count_clone = Arc::clone(&brpop_count);
    let json_clone = envelope_json.clone();
    mock.expect_brpop().returning(move |_, _| {
        let n = brpop_count_clone.fetch_add(1, Ordering::SeqCst);
        let json = json_clone.clone();
        if n == 0 { Ok(Some(json)) } else { Ok(None) }
    });

    mock.expect_load_job_record().returning(|id| {
        let record = test_job_record(id, JobStatus::Queued);
        Ok(Some(record))
    });

    let save_call = Arc::new(AtomicU32::new(0));
    let save_call_clone = Arc::clone(&save_call);
    mock.expect_save_job_record().returning(move |record, _| {
        let call_num = save_call_clone.fetch_add(1, Ordering::SeqCst);
        if call_num == 0 {
            assert_eq!(
                record.status,
                JobStatus::Running,
                "first save_job_record call should set status to Running"
            );
        }
        Ok(())
    });

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move { worker_clone.run().await });

    wait_until(Duration::from_secs(1), || {
        save_call.load(Ordering::SeqCst) >= 1
    })
    .await;
    shutdown_and_join(&worker, handle).await;

    assert!(
        save_call.load(Ordering::SeqCst) >= 1,
        "save_job_record should be called at least once"
    );
}

#[tokio::test]
async fn sets_job_status_to_failed_on_terminal_task_error() {
    let config = test_config();
    let mut mock = MockQueueBackend::new();

    let envelope = test_envelope("job-3", JobType::Index);
    let envelope_json = serde_json::to_string(&envelope).unwrap();

    let brpop_count = Arc::new(AtomicU32::new(0));
    let brpop_count_clone = Arc::clone(&brpop_count);
    let json_clone = envelope_json.clone();
    mock.expect_brpop().returning(move |_, _| {
        let n = brpop_count_clone.fetch_add(1, Ordering::SeqCst);
        let json = json_clone.clone();
        if n == 0 { Ok(Some(json)) } else { Ok(None) }
    });

    mock.expect_load_job_record().returning(|id| {
        let record = test_job_record(id, JobStatus::Queued);
        Ok(Some(record))
    });

    let final_status = Arc::new(Mutex::new(None));
    let final_status_clone = Arc::clone(&final_status);
    mock.expect_save_job_record().returning(move |record, _| {
        *final_status_clone.lock().unwrap() = Some(record.status);
        Ok(())
    });

    mock.expect_lpush().times(0);

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move { worker_clone.run().await });

    wait_until(Duration::from_secs(1), || {
        *final_status.lock().unwrap() == Some(JobStatus::Failed)
    })
    .await;
    shutdown_and_join(&worker, handle).await;

    let status = final_status.lock().unwrap();
    assert_eq!(
        *status,
        Some(JobStatus::Failed),
        "terminal task failures should fail the job instead of retrying blind"
    );
}

#[tokio::test]
async fn does_not_reenqueue_terminal_task_failures() {
    let config = test_config();
    let mut mock = MockQueueBackend::new();

    let envelope = test_envelope("job-4", JobType::Sync);
    let envelope_json = serde_json::to_string(&envelope).unwrap();

    let brpop_count = Arc::new(AtomicU32::new(0));
    let brpop_count_clone = Arc::clone(&brpop_count);
    let json_clone = envelope_json.clone();
    mock.expect_brpop().returning(move |_, _| {
        let n = brpop_count_clone.fetch_add(1, Ordering::SeqCst);
        let json = json_clone.clone();
        if n == 0 { Ok(Some(json)) } else { Ok(None) }
    });

    mock.expect_load_job_record().returning(|id| {
        let mut record = test_job_record(id, JobStatus::Queued);
        record.attempts = 0;
        Ok(Some(record))
    });

    let save_count = Arc::new(AtomicU32::new(0));
    let save_count_clone = Arc::clone(&save_count);
    mock.expect_save_job_record().returning(move |_, _| {
        save_count_clone.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });

    mock.expect_lpush().times(0);

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move { worker_clone.run().await });

    wait_until(Duration::from_secs(1), || {
        save_count.load(Ordering::SeqCst) > 0
    })
    .await;
    shutdown_and_join(&worker, handle).await;
}

#[tokio::test]
async fn marks_job_failed_after_terminal_task_execution() {
    let config = test_config();
    let mut mock = MockQueueBackend::new();

    let envelope = test_envelope("job-5", JobType::Sync);
    let envelope_json = serde_json::to_string(&envelope).unwrap();

    let brpop_count = Arc::new(AtomicU32::new(0));
    let brpop_count_clone = Arc::clone(&brpop_count);
    let json_clone = envelope_json.clone();
    mock.expect_brpop().returning(move |_, _| {
        let n = brpop_count_clone.fetch_add(1, Ordering::SeqCst);
        let json = json_clone.clone();
        if n == 0 { Ok(Some(json)) } else { Ok(None) }
    });

    mock.expect_load_job_record().returning(|id| {
        let mut record = test_job_record(id, JobStatus::Queued);
        record.attempts = 0;
        record.max_retries = 3;
        Ok(Some(record))
    });

    let final_status = Arc::new(Mutex::new(None));
    let final_status_clone = Arc::clone(&final_status);
    mock.expect_save_job_record().returning(move |record, _| {
        *final_status_clone.lock().unwrap() = Some(record.status);
        Ok(())
    });

    mock.expect_lpush().times(0);

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move { worker_clone.run().await });

    wait_until(Duration::from_secs(1), || {
        *final_status.lock().unwrap() == Some(JobStatus::Failed)
    })
    .await;
    shutdown_and_join(&worker, handle).await;

    assert_eq!(
        *final_status.lock().unwrap(),
        Some(JobStatus::Failed),
        "terminal task failures should persist Failed status"
    );
}

// --- Concurrency control ---

#[tokio::test]
async fn respects_max_concurrency_via_semaphore() {
    let mut config = test_config();
    config.max_concurrency = 2;

    let mock = MockQueueBackend::new();
    let worker = Worker::new(config, mock);

    assert_eq!(
        worker.semaphore.available_permits(),
        2,
        "semaphore should have max_concurrency permits"
    );
}

// --- Empty queue handling ---

#[tokio::test]
async fn handles_empty_queue_by_polling_again() {
    let config = test_config();
    let mut mock = MockQueueBackend::new();

    let poll_count = Arc::new(AtomicU32::new(0));
    let poll_count_clone = Arc::clone(&poll_count);
    mock.expect_brpop().returning(move |_, _| {
        poll_count_clone.fetch_add(1, Ordering::SeqCst);
        Ok(None)
    });

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move { worker_clone.run().await });

    wait_until(Duration::from_secs(1), || {
        poll_count.load(Ordering::SeqCst) >= 2
    })
    .await;
    shutdown_and_join(&worker, handle).await;

    let polls = poll_count.load(Ordering::SeqCst);
    assert!(
        polls >= 2,
        "worker should poll multiple times on empty queue, got {polls}"
    );
}

// --- Malformed envelope ---

#[tokio::test]
async fn skips_malformed_envelope_without_crashing() {
    let config = test_config();
    let mut mock = MockQueueBackend::new();

    let brpop_count = Arc::new(AtomicU32::new(0));
    let brpop_count_clone = Arc::clone(&brpop_count);
    mock.expect_brpop().returning(move |_, _| {
        let n = brpop_count_clone.fetch_add(1, Ordering::SeqCst);
        if n == 0 {
            Ok(Some("this is not valid json!!!".to_string()))
        } else {
            Ok(None)
        }
    });

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move { worker_clone.run().await });

    wait_until(Duration::from_secs(1), || {
        brpop_count.load(Ordering::SeqCst) >= 2
    })
    .await;
    shutdown_and_join(&worker, handle).await;
}

// --- Graceful shutdown ---

#[tokio::test]
async fn graceful_shutdown_waits_for_inflight_jobs() {
    let mut config = test_config();
    config.max_concurrency = 1;

    let mut mock = MockQueueBackend::new();

    let envelope = test_envelope("shutdown-job", JobType::Sync);
    let envelope_json = serde_json::to_string(&envelope).unwrap();

    let brpop_count = Arc::new(AtomicU32::new(0));
    let brpop_count_clone = Arc::clone(&brpop_count);
    let json_clone = envelope_json.clone();
    mock.expect_brpop().returning(move |_, _| {
        let n = brpop_count_clone.fetch_add(1, Ordering::SeqCst);
        let json = json_clone.clone();
        if n == 0 { Ok(Some(json)) } else { Ok(None) }
    });

    mock.expect_load_job_record().returning(|id| {
        let record = test_job_record(id, JobStatus::Queued);
        Ok(Some(record))
    });

    let save_count = Arc::new(AtomicU32::new(0));
    let save_count_clone = Arc::clone(&save_count);
    mock.expect_save_job_record().returning(move |_, _| {
        save_count_clone.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });

    mock.expect_lpush().returning(|_, _| Ok(()));

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move { worker_clone.run().await });

    wait_until(Duration::from_secs(1), || {
        save_count.load(Ordering::SeqCst) > 0
    })
    .await;
    shutdown_and_join(&worker, handle).await;

    assert!(
        save_count.load(Ordering::SeqCst) > 0,
        "in-flight job should complete during graceful shutdown"
    );
}

// --- BRPOP uses correct queue name and poll interval as timeout ---

#[tokio::test]
async fn brpop_uses_configured_queue_name_and_timeout() {
    let config = test_config();
    let expected_queue = config.queue_name.clone();
    let expected_timeout = config.poll_interval.as_secs_f64();
    let mut mock = MockQueueBackend::new();

    let brpop_queue = Arc::new(std::sync::Mutex::new(None));
    let brpop_timeout = Arc::new(std::sync::Mutex::new(None));
    let queue_clone = Arc::clone(&brpop_queue);
    let timeout_clone = Arc::clone(&brpop_timeout);

    mock.expect_brpop().returning(move |key, timeout| {
        *queue_clone.lock().unwrap() = Some(key.to_string());
        *timeout_clone.lock().unwrap() = Some(timeout);
        Ok(None)
    });

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move { worker_clone.run().await });

    wait_until(Duration::from_secs(1), || {
        brpop_queue.lock().unwrap().is_some()
    })
    .await;
    shutdown_and_join(&worker, handle).await;

    assert_eq!(
        *brpop_queue.lock().unwrap(),
        Some(expected_queue),
        "brpop should use configured queue_name"
    );
    assert_eq!(
        *brpop_timeout.lock().unwrap(),
        Some(expected_timeout),
        "brpop should use poll_interval as timeout"
    );
}

#[tokio::test]
async fn pg_manager_and_worker_drive_terminal_job_status() {
    let (postgres_url, _container) = match start_postgres_container().await {
        Ok(container) => {
            let host = container.get_host().await.expect("pg host");
            let port = container.get_host_port_ipv4(5432).await.expect("pg port");
            (
                format!("postgres://postgres:postgres@{host}:{port}/postgres"),
                container,
            )
        }
        Err(error) => {
            eprintln!("skipping pg e2e test: {error}");
            return;
        }
    };
    let manager = PgJobManager::new(
        jobs::connection::create_job_pool(&postgres_url)
            .await
            .expect("create pool"),
        3,
        3600,
    )
    .await
    .expect("create job manager");
    let job = manager
        .enqueue_sync_job(
            SyncJobPayload {
                source: "/tmp/collection.anki2".to_string(),
                run_migrations: true,
                index: true,
                reindex_mode: common::ReindexMode::Incremental,
            },
            None,
        )
        .await
        .expect("enqueue sync job");

    let mut config = test_config();
    config.postgres_url = postgres_url.clone();
    config.queue_name = "test:jobs:e2e".to_string();
    config.poll_interval = Duration::from_millis(10);

    let backend = RealPgBackend::connect(&postgres_url)
        .await
        .expect("connect worker backend");
    let worker = Arc::new(Worker::new(config, backend));
    let worker_clone = Arc::clone(&worker);
    let handle = tokio::spawn(async move { worker_clone.run().await });

    wait_until_async(Duration::from_secs(10), || {
        let manager = &manager;
        let job_id = job.job_id.clone();
        async move {
            matches!(
                manager.get_job(&job_id).await,
                Ok(record) if record.status == JobStatus::Failed
            )
        }
    })
    .await;

    let persisted = manager
        .get_job(&job.job_id)
        .await
        .expect("load persisted job");
    assert_eq!(persisted.status, JobStatus::Failed);
    assert!(
        persisted
            .error
            .as_deref()
            .is_some_and(|error| !error.is_empty()),
        "job should persist a concrete runtime failure"
    );

    shutdown_and_join(&worker, handle).await;
}

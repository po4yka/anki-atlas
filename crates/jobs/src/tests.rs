use crate::connection::{parse_redis_url, RedisConfig};
use crate::error::JobError;
use crate::manager::{JobManager, MockJobManager};
use crate::persistence::job_key;
use crate::types::{JobRecord, JobStatus, JobType, JOB_KEY_PREFIX};
use chrono::Utc;
use std::collections::HashMap;

// ── Send + Sync compile-time assertions ──────────────────────────────────────

fn _assert_send_sync<T: Send + Sync>() {}

#[test]
fn all_types_are_send_sync() {
    _assert_send_sync::<JobError>();
    _assert_send_sync::<JobType>();
    _assert_send_sync::<JobStatus>();
    _assert_send_sync::<JobRecord>();
    _assert_send_sync::<RedisConfig>();
}

// ── JobStatus tests ──────────────────────────────────────────────────────────

#[test]
fn job_status_is_terminal_succeeded() {
    assert!(JobStatus::Succeeded.is_terminal());
}

#[test]
fn job_status_is_terminal_failed() {
    assert!(JobStatus::Failed.is_terminal());
}

#[test]
fn job_status_is_terminal_cancelled() {
    assert!(JobStatus::Cancelled.is_terminal());
}

#[test]
fn job_status_is_not_terminal_queued() {
    assert!(!JobStatus::Queued.is_terminal());
}

#[test]
fn job_status_is_not_terminal_scheduled() {
    assert!(!JobStatus::Scheduled.is_terminal());
}

#[test]
fn job_status_is_not_terminal_running() {
    assert!(!JobStatus::Running.is_terminal());
}

#[test]
fn job_status_is_not_terminal_retrying() {
    assert!(!JobStatus::Retrying.is_terminal());
}

#[test]
fn job_status_is_not_terminal_cancel_requested() {
    assert!(!JobStatus::CancelRequested.is_terminal());
}

// ── JobType strum display ────────────────────────────────────────────────────

#[test]
fn job_type_display_sync() {
    assert_eq!(JobType::Sync.to_string(), "sync");
}

#[test]
fn job_type_display_index() {
    assert_eq!(JobType::Index.to_string(), "index");
}

#[test]
fn job_type_from_str() {
    use std::str::FromStr;
    assert_eq!(JobType::from_str("sync").unwrap(), JobType::Sync);
    assert_eq!(JobType::from_str("index").unwrap(), JobType::Index);
    assert!(JobType::from_str("unknown").is_err());
}

// ── JobStatus strum display ──────────────────────────────────────────────────

#[test]
fn job_status_display_queued() {
    assert_eq!(JobStatus::Queued.to_string(), "queued");
}

#[test]
fn job_status_display_cancel_requested() {
    assert_eq!(JobStatus::CancelRequested.to_string(), "cancel_requested");
}

#[test]
fn job_status_from_str() {
    use std::str::FromStr;
    assert_eq!(JobStatus::from_str("queued").unwrap(), JobStatus::Queued);
    assert_eq!(
        JobStatus::from_str("cancel_requested").unwrap(),
        JobStatus::CancelRequested
    );
    assert!(JobStatus::from_str("bogus").is_err());
}

// ── JobRecord JSON roundtrip ─────────────────────────────────────────────────

fn sample_record() -> JobRecord {
    JobRecord {
        job_id: "test-123".to_string(),
        job_type: JobType::Sync,
        status: JobStatus::Queued,
        payload: HashMap::from([("key".to_string(), serde_json::json!("value"))]),
        progress: 42.5,
        message: Some("testing".to_string()),
        attempts: 1,
        max_retries: 3,
        cancel_requested: false,
        created_at: Some(Utc::now()),
        scheduled_for: None,
        started_at: None,
        finished_at: None,
        result: None,
        error: None,
    }
}

#[test]
fn job_record_json_roundtrip() {
    let record = sample_record();
    let json = serde_json::to_string(&record).expect("serialize");
    let restored: JobRecord = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.job_id, record.job_id);
    assert_eq!(restored.job_type, record.job_type);
    assert_eq!(restored.status, record.status);
    assert_eq!(restored.progress, record.progress);
    assert_eq!(restored.message, record.message);
    assert_eq!(restored.attempts, record.attempts);
    assert_eq!(restored.max_retries, record.max_retries);
    assert_eq!(restored.cancel_requested, record.cancel_requested);
}

#[test]
fn job_record_json_with_all_fields() {
    let mut record = sample_record();
    record.started_at = Some(Utc::now());
    record.finished_at = Some(Utc::now());
    record.result = Some(HashMap::from([(
        "count".to_string(),
        serde_json::json!(42),
    )]));
    record.error = Some("some error".to_string());

    let json = serde_json::to_string(&record).expect("serialize");
    let restored: JobRecord = serde_json::from_str(&json).expect("deserialize");
    assert!(restored.started_at.is_some());
    assert!(restored.finished_at.is_some());
    assert!(restored.result.is_some());
    assert_eq!(restored.error.as_deref(), Some("some error"));
}

// ── persistence::job_key ─────────────────────────────────────────────────────

#[test]
fn job_key_format() {
    assert_eq!(job_key("abc-123"), format!("{JOB_KEY_PREFIX}abc-123"));
}

#[test]
fn job_key_produces_correct_prefix() {
    let key = job_key("my-job-id");
    assert_eq!(key, "ankiatlas:job:my-job-id");
}

#[test]
fn job_key_empty_id() {
    assert_eq!(job_key(""), "ankiatlas:job:");
}

// ── connection::parse_redis_url ──────────────────────────────────────────────

#[test]
fn parse_redis_url_basic() {
    let config = parse_redis_url("redis://localhost:6379/0").expect("parse");
    assert_eq!(config.host, "localhost");
    assert_eq!(config.port, 6379);
    assert_eq!(config.database, 0);
    assert!(!config.tls);
    assert!(config.username.is_none());
    assert!(config.password.is_none());
}

#[test]
fn parse_redis_url_with_auth() {
    let config = parse_redis_url("rediss://user:secret@redis.example.com:6380/2").expect("parse");
    assert_eq!(config.host, "redis.example.com");
    assert_eq!(config.port, 6380);
    assert_eq!(config.database, 2);
    assert!(config.tls);
    assert_eq!(config.username.as_deref(), Some("user"));
    assert_eq!(config.password.as_deref(), Some("secret"));
}

#[test]
fn parse_redis_url_default_port() {
    let config = parse_redis_url("redis://myhost/1").expect("parse");
    assert_eq!(config.host, "myhost");
    assert_eq!(config.port, 6379);
    assert_eq!(config.database, 1);
}

#[test]
fn parse_redis_url_no_database() {
    let config = parse_redis_url("redis://localhost").expect("parse");
    assert_eq!(config.database, 0);
}

#[test]
fn parse_redis_url_rejects_http() {
    let result = parse_redis_url("http://localhost:6379/0");
    assert!(result.is_err());
}

#[test]
fn parse_redis_url_rejects_empty() {
    let result = parse_redis_url("");
    assert!(result.is_err());
}

#[test]
fn parse_redis_url_rejects_ftp() {
    let result = parse_redis_url("ftp://localhost/0");
    assert!(result.is_err());
}

// ── MockJobManager compiles ──────────────────────────────────────────────────

#[tokio::test]
async fn mock_job_manager_compiles_and_works() {
    let mut mock = MockJobManager::new();
    mock.expect_get_job()
        .returning(|_| Box::pin(async { Ok(Some(sample_record())) }));

    let result = mock.get_job("test-123").await;
    assert!(result.is_ok());
    let record = result.unwrap().unwrap();
    assert_eq!(record.job_id, "test-123");
}

#[tokio::test]
async fn mock_job_manager_enqueue_sync() {
    let mut mock = MockJobManager::new();
    mock.expect_enqueue_sync_job()
        .returning(|_payload, _run_at| Box::pin(async { Ok(sample_record()) }));

    let result = mock
        .enqueue_sync_job(HashMap::new(), None)
        .await
        .expect("enqueue");
    assert_eq!(result.job_type, JobType::Sync);
}

#[tokio::test]
async fn mock_job_manager_enqueue_index() {
    let mut mock = MockJobManager::new();
    mock.expect_enqueue_index_job()
        .returning(|_payload, _run_at| {
            Box::pin(async {
                let mut rec = sample_record();
                rec.job_type = JobType::Index;
                Ok(rec)
            })
        });

    let result = mock
        .enqueue_index_job(HashMap::new(), None)
        .await
        .expect("enqueue");
    assert_eq!(result.job_type, JobType::Index);
}

#[tokio::test]
async fn mock_job_manager_cancel() {
    let mut mock = MockJobManager::new();
    mock.expect_cancel_job().returning(|_| {
        Box::pin(async {
            let mut rec = sample_record();
            rec.status = JobStatus::Cancelled;
            Ok(Some(rec))
        })
    });

    let result = mock.cancel_job("test-123").await.expect("cancel");
    assert_eq!(result.unwrap().status, JobStatus::Cancelled);
}

#[tokio::test]
async fn mock_job_manager_close() {
    let mut mock = MockJobManager::new();
    mock.expect_close()
        .returning(|| Box::pin(async { Ok(()) }));

    mock.close().await.expect("close");
}

// ── Error display ────────────────────────────────────────────────────────────

#[test]
fn error_display_messages() {
    let e = JobError::BackendUnavailable("conn refused".to_string());
    assert!(e.to_string().contains("conn refused"));

    let e = JobError::NotFound("job-42".to_string());
    assert!(e.to_string().contains("job-42"));

    let e = JobError::TerminalState {
        job_id: "j1".to_string(),
        status: "succeeded".to_string(),
    };
    assert!(e.to_string().contains("succeeded"));

    let e = JobError::Redis("timeout".to_string());
    assert!(e.to_string().contains("timeout"));

    let e = JobError::Serialization("bad json".to_string());
    assert!(e.to_string().contains("bad json"));

    let e = JobError::TaskExecution("panic".to_string());
    assert!(e.to_string().contains("panic"));
}

// ── JOB_KEY_PREFIX constant ──────────────────────────────────────────────────

#[test]
fn job_key_prefix_value() {
    assert_eq!(JOB_KEY_PREFIX, "ankiatlas:job:");
}

use crate::error::JobError;
use crate::manager::{JobManager, MockJobManager};
use crate::types::{
    IndexJobPayload, JobPayload, JobRecord, JobResultData, JobStatus, JobType, SyncJobPayload,
    SyncJobResult,
};
use chrono::Utc;

// ── Send + Sync compile-time assertions ──────────────────────────────────────

fn _assert_send_sync<T: Send + Sync>() {}

#[test]
fn all_types_are_send_sync() {
    _assert_send_sync::<JobError>();
    _assert_send_sync::<JobType>();
    _assert_send_sync::<JobStatus>();
    _assert_send_sync::<JobRecord>();
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
        payload: JobPayload::Sync(SyncJobPayload {
            source: "/tmp/collection.anki2".to_string(),
            run_migrations: true,
            index: true,
            reindex_mode: common::ReindexMode::Incremental,
        }),
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
    record.result = Some(JobResultData::Sync(SyncJobResult {
        decks_upserted: 1,
        models_upserted: 2,
        notes_upserted: 3,
        notes_deleted: 0,
        cards_upserted: 4,
        card_stats_upserted: 5,
        duration_ms: 250,
        notes_embedded: Some(6),
        notes_skipped: Some(1),
        index_errors: vec!["minor warning".to_string()],
    }));
    record.error = Some("some error".to_string());

    let json = serde_json::to_string(&record).expect("serialize");
    let restored: JobRecord = serde_json::from_str(&json).expect("deserialize");
    assert!(restored.started_at.is_some());
    assert!(restored.finished_at.is_some());
    assert!(restored.result.is_some());
    assert_eq!(restored.error.as_deref(), Some("some error"));
}

// ── MockJobManager compiles ──────────────────────────────────────────────────

#[tokio::test]
async fn mock_job_manager_compiles_and_works() {
    let mut mock = MockJobManager::new();
    mock.expect_get_job()
        .returning(|_| Box::pin(async { Ok(sample_record()) }));

    let result = mock.get_job("test-123").await;
    assert!(result.is_ok());
    let record = result.unwrap();
    assert_eq!(record.job_id, "test-123");
}

#[tokio::test]
async fn mock_job_manager_enqueue_sync() {
    let mut mock = MockJobManager::new();
    mock.expect_enqueue_sync_job()
        .returning(|_payload, _run_at| Box::pin(async { Ok(sample_record()) }));

    let result = mock
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
                rec.payload = JobPayload::Index(IndexJobPayload {
                    reindex_mode: common::ReindexMode::Incremental,
                });
                Ok(rec)
            })
        });

    let result = mock
        .enqueue_index_job(
            IndexJobPayload {
                reindex_mode: common::ReindexMode::Incremental,
            },
            None,
        )
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
            Ok(rec)
        })
    });

    let result = mock.cancel_job("test-123").await.expect("cancel");
    assert_eq!(result.status, JobStatus::Cancelled);
}

#[tokio::test]
async fn mock_job_manager_close() {
    let mut mock = MockJobManager::new();
    mock.expect_close().returning(|| Box::pin(async { Ok(()) }));

    mock.close().await.expect("close");
}

#[test]
fn unsupported_error_is_not_retryable() {
    let error = JobError::Unsupported("scheduled jobs are not supported yet".to_string());
    assert!(!error.is_retryable());
}

#[test]
fn database_error_is_retryable() {
    let error = JobError::Database("timeout".to_string());
    assert!(error.is_retryable());
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

    let e = JobError::Database("timeout".to_string());
    assert!(e.to_string().contains("timeout"));

    let e = JobError::Serialization("bad json".to_string());
    assert!(e.to_string().contains("bad json"));

    let e = JobError::TaskExecution("panic".to_string());
    assert!(e.to_string().contains("panic"));

    let e = JobError::Unsupported("scheduling".to_string());
    assert!(e.to_string().contains("scheduling"));
}

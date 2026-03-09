use super::*;
use crate::config::WorkerConfig;
use crate::envelope::JobEnvelope;
use jobs::types::{JobRecord, JobStatus, JobType};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

fn test_config() -> WorkerConfig {
    WorkerConfig {
        redis_url: "redis://localhost:6379".to_string(),
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
        payload: HashMap::new(),
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
    JobEnvelope {
        job_id: job_id.to_string(),
        job_type,
        payload: HashMap::new(),
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

    mock.expect_brpop()
        .returning(|_, _| Box::pin(async { Ok(None) }));

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let run_handle = tokio::spawn(async move {
        worker_clone.run().await
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    assert!(!run_handle.is_finished(), "run() should block until shutdown");

    worker.shutdown(Duration::from_secs(1)).await;

    let result = tokio::time::timeout(Duration::from_secs(2), run_handle).await;
    assert!(result.is_ok(), "run() should complete after shutdown signal");
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
    mock.expect_brpop()
        .returning(move |_, _| {
            let n = call_count_clone.fetch_add(1, Ordering::SeqCst);
            let json = json_clone.clone();
            Box::pin(async move {
                if n == 0 {
                    Ok(Some(json))
                } else {
                    Ok(None)
                }
            })
        });

    mock.expect_load_job_record()
        .returning(|id| {
            let record = test_job_record(id, JobStatus::Queued);
            Box::pin(async { Ok(Some(record)) })
        });

    mock.expect_save_job_record()
        .returning(|_, _| Box::pin(async { Ok(()) }));

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move {
        worker_clone.run().await
    });

    tokio::time::sleep(Duration::from_millis(200)).await;
    worker.shutdown(Duration::from_secs(1)).await;
    let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
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
    mock.expect_brpop()
        .returning(move |_, _| {
            let n = brpop_count_clone.fetch_add(1, Ordering::SeqCst);
            let json = json_clone.clone();
            Box::pin(async move {
                if n == 0 { Ok(Some(json)) } else { Ok(None) }
            })
        });

    mock.expect_load_job_record()
        .returning(|id| {
            let record = test_job_record(id, JobStatus::Queued);
            Box::pin(async { Ok(Some(record)) })
        });

    let save_call = Arc::new(AtomicU32::new(0));
    let save_call_clone = Arc::clone(&save_call);
    mock.expect_save_job_record()
        .returning(move |record, _| {
            let call_num = save_call_clone.fetch_add(1, Ordering::SeqCst);
            let status = record.status;
            Box::pin(async move {
                if call_num == 0 {
                    assert_eq!(
                        status,
                        JobStatus::Running,
                        "first save_job_record call should set status to Running"
                    );
                }
                Ok(())
            })
        });

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move {
        worker_clone.run().await
    });

    tokio::time::sleep(Duration::from_millis(200)).await;
    worker.shutdown(Duration::from_secs(1)).await;
    let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;

    assert!(save_call.load(Ordering::SeqCst) >= 1, "save_job_record should be called at least once");
}

#[tokio::test]
async fn sets_job_status_to_retrying_on_first_failure() {
    let config = test_config();
    let mut mock = MockQueueBackend::new();

    let envelope = test_envelope("job-3", JobType::Index);
    let envelope_json = serde_json::to_string(&envelope).unwrap();

    let brpop_count = Arc::new(AtomicU32::new(0));
    let brpop_count_clone = Arc::clone(&brpop_count);
    let json_clone = envelope_json.clone();
    mock.expect_brpop()
        .returning(move |_, _| {
            let n = brpop_count_clone.fetch_add(1, Ordering::SeqCst);
            let json = json_clone.clone();
            Box::pin(async move {
                if n == 0 { Ok(Some(json)) } else { Ok(None) }
            })
        });

    mock.expect_load_job_record()
        .returning(|id| {
            let record = test_job_record(id, JobStatus::Queued);
            Box::pin(async { Ok(Some(record)) })
        });

    let final_status = Arc::new(std::sync::Mutex::new(None));
    let final_status_clone = Arc::clone(&final_status);
    mock.expect_save_job_record()
        .returning(move |record, _| {
            let status = record.status;
            let fs = Arc::clone(&final_status_clone);
            Box::pin(async move {
                *fs.lock().unwrap() = Some(status);
                Ok(())
            })
        });

    mock.expect_lpush()
        .returning(|_, _| Box::pin(async { Ok(()) }));

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move {
        worker_clone.run().await
    });

    tokio::time::sleep(Duration::from_millis(200)).await;
    worker.shutdown(Duration::from_secs(1)).await;
    let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;

    let status = final_status.lock().unwrap();
    assert_eq!(
        *status,
        Some(JobStatus::Retrying),
        "first failure with retries remaining should result in Retrying status"
    );
}

#[tokio::test]
async fn reenqueues_job_on_retryable_error_below_max_retries() {
    let config = test_config();
    let mut mock = MockQueueBackend::new();

    let envelope = test_envelope("job-4", JobType::Sync);
    let envelope_json = serde_json::to_string(&envelope).unwrap();

    let brpop_count = Arc::new(AtomicU32::new(0));
    let brpop_count_clone = Arc::clone(&brpop_count);
    let json_clone = envelope_json.clone();
    mock.expect_brpop()
        .returning(move |_, _| {
            let n = brpop_count_clone.fetch_add(1, Ordering::SeqCst);
            let json = json_clone.clone();
            Box::pin(async move {
                if n == 0 { Ok(Some(json)) } else { Ok(None) }
            })
        });

    mock.expect_load_job_record()
        .returning(|id| {
            let mut record = test_job_record(id, JobStatus::Queued);
            record.attempts = 0;
            Box::pin(async { Ok(Some(record)) })
        });

    mock.expect_save_job_record()
        .returning(|_, _| Box::pin(async { Ok(()) }));

    let lpush_called = Arc::new(AtomicU32::new(0));
    let lpush_called_clone = Arc::clone(&lpush_called);
    mock.expect_lpush()
        .returning(move |_, _| {
            lpush_called_clone.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(()) })
        });

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move {
        worker_clone.run().await
    });

    tokio::time::sleep(Duration::from_millis(200)).await;
    worker.shutdown(Duration::from_secs(1)).await;
    let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;

    assert!(
        lpush_called.load(Ordering::SeqCst) > 0,
        "should re-enqueue job when attempts < max_retries"
    );
}

#[tokio::test]
async fn does_not_reenqueue_after_max_retries_exhausted() {
    let config = test_config();
    let mut mock = MockQueueBackend::new();

    let envelope = test_envelope("job-5", JobType::Sync);
    let envelope_json = serde_json::to_string(&envelope).unwrap();

    let brpop_count = Arc::new(AtomicU32::new(0));
    let brpop_count_clone = Arc::clone(&brpop_count);
    let json_clone = envelope_json.clone();
    mock.expect_brpop()
        .returning(move |_, _| {
            let n = brpop_count_clone.fetch_add(1, Ordering::SeqCst);
            let json = json_clone.clone();
            Box::pin(async move {
                if n == 0 { Ok(Some(json)) } else { Ok(None) }
            })
        });

    mock.expect_load_job_record()
        .returning(|id| {
            let mut record = test_job_record(id, JobStatus::Queued);
            record.attempts = 3;
            record.max_retries = 3;
            Box::pin(async { Ok(Some(record)) })
        });

    let final_status = Arc::new(std::sync::Mutex::new(None));
    let final_status_clone = Arc::clone(&final_status);
    mock.expect_save_job_record()
        .returning(move |record, _| {
            let status = record.status;
            let fs = Arc::clone(&final_status_clone);
            Box::pin(async move {
                *fs.lock().unwrap() = Some(status);
                Ok(())
            })
        });

    let lpush_called = Arc::new(AtomicU32::new(0));
    let lpush_called_clone = Arc::clone(&lpush_called);
    mock.expect_lpush()
        .returning(move |_, _| {
            lpush_called_clone.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(()) })
        });

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move {
        worker_clone.run().await
    });

    tokio::time::sleep(Duration::from_millis(200)).await;
    worker.shutdown(Duration::from_secs(1)).await;
    let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;

    assert_eq!(
        lpush_called.load(Ordering::SeqCst),
        0,
        "should NOT re-enqueue when max_retries exhausted"
    );
    assert_eq!(
        *final_status.lock().unwrap(),
        Some(JobStatus::Failed),
        "should set Failed status when max_retries exhausted"
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
    mock.expect_brpop()
        .returning(move |_, _| {
            poll_count_clone.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(None) })
        });

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move {
        worker_clone.run().await
    });

    tokio::time::sleep(Duration::from_millis(200)).await;
    worker.shutdown(Duration::from_secs(1)).await;
    let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;

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
    mock.expect_brpop()
        .returning(move |_, _| {
            let n = brpop_count_clone.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                if n == 0 {
                    Ok(Some("this is not valid json!!!".to_string()))
                } else {
                    Ok(None)
                }
            })
        });

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move {
        worker_clone.run().await
    });

    tokio::time::sleep(Duration::from_millis(150)).await;
    worker.shutdown(Duration::from_secs(1)).await;

    let result = tokio::time::timeout(Duration::from_secs(2), handle).await;
    assert!(
        result.is_ok(),
        "worker should not crash on malformed envelope"
    );
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
    mock.expect_brpop()
        .returning(move |_, _| {
            let n = brpop_count_clone.fetch_add(1, Ordering::SeqCst);
            let json = json_clone.clone();
            Box::pin(async move {
                if n == 0 { Ok(Some(json)) } else { Ok(None) }
            })
        });

    mock.expect_load_job_record()
        .returning(|id| {
            let record = test_job_record(id, JobStatus::Queued);
            Box::pin(async { Ok(Some(record)) })
        });

    let save_count = Arc::new(AtomicU32::new(0));
    let save_count_clone = Arc::clone(&save_count);
    mock.expect_save_job_record()
        .returning(move |_, _| {
            save_count_clone.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(()) })
        });

    mock.expect_lpush()
        .returning(|_, _| Box::pin(async { Ok(()) }));

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move {
        worker_clone.run().await
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    worker.shutdown(Duration::from_secs(5)).await;

    let result = tokio::time::timeout(Duration::from_secs(3), handle).await;
    assert!(result.is_ok(), "shutdown should complete after in-flight jobs finish");

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

    mock.expect_brpop()
        .returning(move |key, timeout| {
            *queue_clone.lock().unwrap() = Some(key.to_string());
            *timeout_clone.lock().unwrap() = Some(timeout);
            Box::pin(async { Ok(None) })
        });

    let worker = Arc::new(Worker::new(config, mock));
    let worker_clone = Arc::clone(&worker);

    let handle = tokio::spawn(async move {
        worker_clone.run().await
    });

    tokio::time::sleep(Duration::from_millis(100)).await;
    worker.shutdown(Duration::from_secs(1)).await;
    let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;

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

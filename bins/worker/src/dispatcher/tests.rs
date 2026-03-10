use super::*;
use crate::envelope::JobEnvelope;
use jobs::{IndexJobPayload, JobPayload, JobType, SyncJobPayload};

/// Helper to build a test envelope.
fn make_envelope(job_type: JobType) -> JobEnvelope {
    let payload = match job_type {
        JobType::Sync => JobPayload::Sync(SyncJobPayload {
            source: "/tmp/collection.anki2".to_string(),
            run_migrations: true,
            index: true,
            force_reindex: false,
        }),
        JobType::Index => JobPayload::Index(IndexJobPayload {
            force_reindex: false,
        }),
    };

    JobEnvelope {
        job_id: "test-job-1".to_string(),
        job_type,
        payload,
    }
}

#[test]
fn types_are_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<JobEnvelope>();
    assert_sync::<JobEnvelope>();
    assert_send::<crate::config::WorkerConfig>();
    assert_sync::<crate::config::WorkerConfig>();
}

#[test]
fn dispatch_function_exists() {
    // Verify the function is accessible and compiles with the correct signature.
    let _f = dispatch;
    let _ = make_envelope(JobType::Sync);
    let _ = make_envelope(JobType::Index);
}

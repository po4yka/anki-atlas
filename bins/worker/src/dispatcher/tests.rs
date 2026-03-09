use super::*;
use crate::envelope::JobEnvelope;
use std::collections::HashMap;

/// Helper to build a test envelope.
fn make_envelope(job_type: JobType) -> JobEnvelope {
    JobEnvelope {
        job_id: "test-job-1".to_string(),
        job_type,
        payload: HashMap::new(),
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
}

use super::*;
use crate::envelope::JobEnvelope;
use std::collections::HashMap;

/// Helper to build a test envelope.
fn make_envelope(task_name: &str) -> JobEnvelope {
    JobEnvelope {
        job_id: "test-job-1".to_string(),
        task_name: task_name.to_string(),
        payload: HashMap::new(),
    }
}

// Verify Send + Sync bounds on key types
#[test]
fn types_are_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<JobEnvelope>();
    assert_sync::<JobEnvelope>();
}

// Verify dispatch exists and compiles - will panic with todo!() in RED phase.
// Once implemented, these should test actual routing behavior.
#[test]
fn dispatch_function_exists() {
    // Just verify the function is accessible and has correct module path.
    // Cannot call it without a TaskContext (requires Redis), but we confirm it compiles.
    let _f = dispatch;
    let _ = make_envelope("job_sync");
}

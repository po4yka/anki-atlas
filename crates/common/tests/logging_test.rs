use common::logging::*;
use std::sync::{Arc, Mutex};

// ── Helper: capture writer ──────────────────────────────────────────────

/// A thread-safe in-memory writer for capturing log output.
#[derive(Clone)]
struct CaptureWriter {
    buf: Arc<Mutex<Vec<u8>>>,
}

impl CaptureWriter {
    fn new() -> Self {
        Self {
            buf: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn contents(&self) -> String {
        let buf = self.buf.lock().unwrap();
        String::from_utf8_lossy(&buf).to_string()
    }
}

impl std::io::Write for CaptureWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buf.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// ── Correlation ID ──────────────────────────────────────────────────────

#[test]
fn correlation_id_initially_none() {
    // Before setting, get_correlation_id should return None
    clear_correlation_id();
    assert!(get_correlation_id().is_none());
}

#[test]
fn set_correlation_id_with_explicit_value() {
    // Setting with Some(id) should store and return that id
    let id = set_correlation_id(Some("req-123".to_string()));
    assert_eq!(id, "req-123");
    assert_eq!(get_correlation_id(), Some("req-123".to_string()));
    clear_correlation_id();
}

#[test]
fn set_correlation_id_generates_uuid_when_none() {
    // Setting with None should auto-generate a UUID
    let id = set_correlation_id(None);
    assert!(!id.is_empty(), "generated ID should not be empty");
    assert!(id.len() >= 32, "generated ID should look like a UUID");
    assert_eq!(get_correlation_id(), Some(id.clone()));
    clear_correlation_id();
}

#[test]
fn clear_correlation_id_removes_it() {
    set_correlation_id(Some("to-be-cleared".to_string()));
    assert!(get_correlation_id().is_some());
    clear_correlation_id();
    assert!(get_correlation_id().is_none());
}

#[test]
fn set_correlation_id_overwrites_previous() {
    set_correlation_id(Some("first".to_string()));
    set_correlation_id(Some("second".to_string()));
    assert_eq!(get_correlation_id(), Some("second".to_string()));
    clear_correlation_id();
}

#[test]
fn set_correlation_id_returns_the_value_set() {
    // The return value should match what was actually stored
    let returned = set_correlation_id(Some("abc-def".to_string()));
    let stored = get_correlation_id().unwrap();
    assert_eq!(returned, stored);
    clear_correlation_id();
}

// ── configure_logging: JSON output ──────────────────────────────────────

#[test]
fn configure_logging_json_produces_json_lines() {
    // When json_output=true, output should be valid JSON
    let writer = CaptureWriter::new();
    let reader = writer.clone();

    configure_logging(false, true, writer);

    // Emit a tracing event
    tracing::info!(key = "value", "test message");

    let output = reader.contents();
    assert!(!output.is_empty(), "JSON logging should produce output");

    // Each line should be parseable as JSON
    for line in output.lines() {
        let parsed: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("line is not valid JSON: {e}\nline: {line}"));
        // Should contain the message
        let msg = parsed.get("fields").and_then(|f| f.get("message"));
        assert!(
            msg.is_some() || parsed.get("message").is_some(),
            "JSON log line should contain message field"
        );
    }
}

#[test]
fn configure_logging_debug_level_emits_debug_events() {
    // When debug=true, DEBUG level events should appear
    let writer = CaptureWriter::new();
    let reader = writer.clone();

    configure_logging(true, true, writer);

    tracing::debug!("debug-level-test");

    let output = reader.contents();
    assert!(
        output.contains("debug-level-test"),
        "debug=true should capture DEBUG events, got: {output}"
    );
}

#[test]
fn configure_logging_info_level_omits_debug_events() {
    // When debug=false, DEBUG level events should NOT appear
    let writer = CaptureWriter::new();
    let reader = writer.clone();

    configure_logging(false, true, writer);

    tracing::debug!("should-not-appear");
    tracing::info!("should-appear");

    let output = reader.contents();
    assert!(
        !output.contains("should-not-appear"),
        "debug=false should omit DEBUG events"
    );
    assert!(
        output.contains("should-appear"),
        "debug=false should still capture INFO events"
    );
}

// ── configure_logging: human-readable output ────────────────────────────

#[test]
fn configure_logging_human_readable_is_not_json() {
    // When json_output=false, output should NOT be JSON
    let writer = CaptureWriter::new();
    let reader = writer.clone();

    configure_logging(false, false, writer);

    tracing::info!("human-readable-test");

    let output = reader.contents();
    assert!(
        !output.is_empty(),
        "human-readable logging should produce output"
    );

    // Should NOT be valid JSON (human-readable format)
    for line in output.lines() {
        assert!(
            serde_json::from_str::<serde_json::Value>(line).is_err(),
            "human-readable output should not be JSON: {line}"
        );
    }
}

// ── Correlation ID in logs ──────────────────────────────────────────────

#[test]
fn correlation_id_appears_in_json_log_output() {
    // After setting a correlation ID, it should appear in log output
    let writer = CaptureWriter::new();
    let reader = writer.clone();

    configure_logging(false, true, writer);

    set_correlation_id(Some("corr-42".to_string()));
    tracing::info!("correlated-event");

    let output = reader.contents();
    assert!(
        output.contains("corr-42"),
        "correlation ID should appear in log output, got: {output}"
    );

    clear_correlation_id();
}

// ── Crate root re-exports ───────────────────────────────────────────────

#[test]
fn crate_reexports_logging_functions() {
    // Verify logging functions are accessible from crate root or logging module
    let _ = common::logging::get_correlation_id;
    let _ = common::logging::set_correlation_id;
    let _ = common::logging::clear_correlation_id;
    // configure_logging takes impl Write, verify it's callable with a concrete type
    configure_logging(false, false, std::io::sink());
}

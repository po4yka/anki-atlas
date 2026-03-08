use std::io;

/// Initialize the global tracing subscriber.
/// Call once per process entry point.
///
/// - `debug`: if true, set level to DEBUG; otherwise INFO.
/// - `json_output`: if true, emit JSON lines; otherwise human-readable.
/// - `writer`: output destination (defaults to stderr).
pub fn configure_logging(
    _debug: bool,
    _json_output: bool,
    _writer: impl io::Write + Send + 'static,
) {
    // TODO(ralph): implement
}

/// Correlation ID stored in a task-local (tokio) or thread-local.
/// Returns `None` if not set.
pub fn get_correlation_id() -> Option<String> {
    None
}

/// Set or generate a correlation ID. Returns the ID that was set.
pub fn set_correlation_id(_id: Option<String>) -> String {
    String::new()
}

/// Clear the correlation ID.
pub fn clear_correlation_id() {}

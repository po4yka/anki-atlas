use std::cell::RefCell;
use std::io;

use tracing::Subscriber;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{EnvFilter, Layer};
use uuid::Uuid;

// ── Correlation ID (thread-local) ──────────────────────────────────────

thread_local! {
    static CORRELATION_ID: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Returns the current correlation ID, or `None` if not set.
pub fn get_correlation_id() -> Option<String> {
    CORRELATION_ID.with(|cell| cell.borrow().clone())
}

/// Set or generate a correlation ID. Returns the ID that was set.
///
/// - `Some(id)` stores that exact ID.
/// - `None` generates a new UUID v4.
pub fn set_correlation_id(id: Option<String>) -> String {
    let value = id.unwrap_or_else(|| Uuid::new_v4().to_string());
    CORRELATION_ID.with(|cell| {
        *cell.borrow_mut() = Some(value.clone());
    });
    value
}

/// Clear the correlation ID.
pub fn clear_correlation_id() {
    CORRELATION_ID.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

// ── Logging configuration ──────────────────────────────────────────────

/// A wrapper that makes a cloneable writer usable as a `MakeWriter`.
#[derive(Clone)]
struct WriterMaker<W: io::Write + Send + Sync + Clone + 'static>(W);

impl<W: io::Write + Send + Sync + Clone + 'static> MakeWriter<'_> for WriterMaker<W> {
    type Writer = W;

    fn make_writer(&self) -> Self::Writer {
        self.0.clone()
    }
}

/// Build a subscriber and install it as the thread-local default.
///
/// The guard is intentionally leaked so the subscriber remains active
/// for the lifetime of the thread.
fn install_subscriber(subscriber: impl Subscriber + Send + Sync + 'static) {
    let guard = tracing::subscriber::set_default(subscriber);
    std::mem::forget(guard);
}

/// A formatting layer that appends the correlation ID to each log line.
/// Works by wrapping the inner writer and injecting the correlation ID
/// into the JSON output.
struct CorrelationWriter<W: io::Write> {
    inner: W,
}

impl<W: io::Write> io::Write for CorrelationWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let len = buf.len();
        if let Some(corr_id) = get_correlation_id() {
            // Try to inject correlation_id into JSON lines
            let s = String::from_utf8_lossy(buf);
            for line in s.lines() {
                if let Ok(mut val) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(obj) = val.as_object_mut() {
                        obj.insert(
                            "correlation_id".to_string(),
                            serde_json::Value::String(corr_id.clone()),
                        );
                    }
                    let mut out = serde_json::to_string(&val).unwrap_or_else(|_| line.to_string());
                    out.push('\n');
                    self.inner.write_all(out.as_bytes())?;
                } else {
                    // Non-JSON line: append correlation_id as text
                    let mut line = line.to_string();
                    line.push_str(&format!(" correlation_id={corr_id}"));
                    line.push('\n');
                    self.inner.write_all(line.as_bytes())?;
                }
            }
        } else {
            self.inner.write_all(buf)?;
        }
        Ok(len)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// A MakeWriter that wraps an inner MakeWriter's writers with CorrelationWriter.
#[derive(Clone)]
struct CorrelationMaker<M>(M);

impl<'a, M> MakeWriter<'a> for CorrelationMaker<M>
where
    M: MakeWriter<'a>,
{
    type Writer = CorrelationWriter<M::Writer>;

    fn make_writer(&'a self) -> Self::Writer {
        CorrelationWriter {
            inner: self.0.make_writer(),
        }
    }
}

/// Initialize the tracing subscriber for the current thread.
///
/// - `debug`: if true, set level to DEBUG; otherwise INFO.
/// - `json_output`: if true, emit JSON lines; otherwise human-readable.
/// - `writer`: output destination (defaults to stderr in production).
pub fn configure_logging(
    debug: bool,
    json_output: bool,
    writer: impl io::Write + Send + Sync + Clone + 'static,
) {
    let filter = if debug {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };
    let maker = CorrelationMaker(WriterMaker(writer));

    if json_output {
        let layer = tracing_subscriber::fmt::layer()
            .json()
            .with_writer(maker)
            .with_filter(filter);
        let subscriber = tracing_subscriber::registry().with(layer);
        install_subscriber(subscriber);
    } else {
        let layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(maker)
            .with_filter(filter);
        let subscriber = tracing_subscriber::registry().with(layer);
        install_subscriber(subscriber);
    };
}

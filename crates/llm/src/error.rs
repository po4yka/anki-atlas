use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("provider error: {message}")]
    Provider {
        message: String,
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("invalid JSON response: {message}")]
    InvalidJson {
        message: String,
        response_text: String,
    },

    #[error("request failed after {attempts} retries: {message}")]
    Exhausted { message: String, attempts: u32 },

    #[error("HTTP error {status}: {body}")]
    Http { status: u16, body: String },

    #[error("connection error: {0}")]
    Connection(String),
}

#[cfg(test)]
mod tests;

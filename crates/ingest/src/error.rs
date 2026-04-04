use thiserror::Error;

#[derive(Debug, Error)]
pub enum IngestError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("PDF extraction failed: {0}")]
    Pdf(String),

    #[error("web fetch failed: {0}")]
    Web(String),

    #[error("content extraction failed: {0}")]
    Extraction(String),

    #[error("note write failed: {0}")]
    NoteWrite(String),
}

impl From<reqwest::Error> for IngestError {
    fn from(e: reqwest::Error) -> Self {
        Self::Web(e.to_string())
    }
}

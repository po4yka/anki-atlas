use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Source of content to ingest.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IngestSource {
    Pdf { path: PathBuf },
    WebPage { url: String },
    TextFile { path: PathBuf },
}

/// An image extracted from the source document.
#[derive(Debug, Clone)]
pub struct ExtractedImage {
    pub filename: String,
    pub data: Vec<u8>,
    pub mime_type: String,
    pub caption: Option<String>,
}

/// Extracted document ready for note creation and card generation.
#[derive(Debug, Clone)]
pub struct IngestedDocument {
    pub title: String,
    pub body_markdown: String,
    pub images: Vec<ExtractedImage>,
    pub source_url: Option<String>,
    pub source_path: Option<PathBuf>,
    pub metadata: HashMap<String, String>,
}

/// Result of a full ingest + generate pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResult {
    pub note_path: PathBuf,
    pub document_title: String,
    pub sections_found: usize,
    pub images_found: usize,
    pub source_type: String,
}

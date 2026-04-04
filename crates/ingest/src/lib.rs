pub mod error;
pub mod models;
pub mod note_writer;
pub mod pdf;
pub mod pipeline;
pub mod text;
pub mod web;

pub use error::IngestError;
pub use models::{ExtractedImage, IngestResult, IngestSource, IngestedDocument};
pub use pipeline::{extract_document, ingest};

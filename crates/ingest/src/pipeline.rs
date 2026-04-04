use std::path::Path;

use tracing::instrument;

use crate::error::IngestError;
use crate::models::{IngestResult, IngestSource, IngestedDocument};
use crate::{note_writer, pdf, text, web};

/// Ingest content from any supported source and write an Obsidian note.
///
/// Returns the path to the created note and metadata about the ingestion.
#[instrument(skip(vault_root))]
pub async fn ingest(
    source: &IngestSource,
    vault_root: &Path,
    target_folder: &str,
) -> Result<IngestResult, IngestError> {
    let (doc, source_type) = extract_document(source).await?;

    let sections_found = doc
        .body_markdown
        .lines()
        .filter(|l| l.starts_with('#'))
        .count();
    let images_found = doc.images.len();
    let document_title = doc.title.clone();

    let note_path = note_writer::write_obsidian_note(&doc, vault_root, target_folder)?;

    Ok(IngestResult {
        note_path,
        document_title,
        sections_found,
        images_found,
        source_type,
    })
}

/// Extract an IngestedDocument from a source without writing a note.
pub async fn extract_document(
    source: &IngestSource,
) -> Result<(IngestedDocument, String), IngestError> {
    match source {
        IngestSource::Pdf { path } => Ok((pdf::ingest_pdf(path)?, "pdf".to_string())),
        IngestSource::WebPage { url } => Ok((web::ingest_web_page(url).await?, "web".to_string())),
        IngestSource::TextFile { path } => Ok((text::ingest_text_file(path)?, "text".to_string())),
    }
}

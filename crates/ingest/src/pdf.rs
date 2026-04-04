use std::path::Path;

use crate::error::IngestError;
use crate::models::IngestedDocument;

/// Ingest a PDF file by extracting its text content.
pub fn ingest_pdf(path: &Path) -> Result<IngestedDocument, IngestError> {
    let bytes = std::fs::read(path)?;
    let text = pdf_extract::extract_text_from_mem(&bytes)
        .map_err(|e| IngestError::Pdf(format!("text extraction failed: {e}")))?;

    if text.trim().is_empty() {
        return Err(IngestError::Pdf(
            "PDF contains no extractable text (may be image-only)".into(),
        ));
    }

    let title = extract_title(&text, path);
    let body_markdown = text_to_markdown(&text);

    Ok(IngestedDocument {
        title,
        body_markdown,
        images: vec![], // PDF image extraction requires more complex libraries
        source_path: Some(path.to_path_buf()),
        source_url: None,
        metadata: Default::default(),
    })
}

/// Extract title from first non-empty line or fall back to filename.
fn extract_title(text: &str, path: &Path) -> String {
    text.lines()
        .find(|line| !line.trim().is_empty())
        .map(|line| line.trim().to_string())
        .filter(|title| title.len() <= 200)
        .unwrap_or_else(|| {
            path.file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string()
        })
}

/// Convert raw PDF text to markdown by adding paragraph breaks.
fn text_to_markdown(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_empty = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !prev_empty {
                result.push_str("\n\n");
                prev_empty = true;
            }
        } else {
            if !prev_empty && !result.is_empty() {
                result.push(' ');
            }
            result.push_str(trimmed);
            prev_empty = false;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_to_markdown_joins_lines() {
        let input = "Hello\nworld\n\nNew paragraph";
        let result = text_to_markdown(input);
        assert!(result.contains("Hello world"));
        assert!(result.contains("\n\nNew paragraph"));
    }

    #[test]
    fn extract_title_from_first_line() {
        let title = extract_title("Introduction to Rust\n\nMore text", Path::new("doc.pdf"));
        assert_eq!(title, "Introduction to Rust");
    }
}

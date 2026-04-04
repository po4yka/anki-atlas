use std::path::Path;

use crate::error::IngestError;
use crate::models::IngestedDocument;

/// Ingest a plain text or markdown file.
pub fn ingest_text_file(path: &Path) -> Result<IngestedDocument, IngestError> {
    let content = std::fs::read_to_string(path)?;
    let is_markdown = path
        .extension()
        .is_some_and(|e| e == "md" || e == "markdown" || e == "txt");

    let (title, body) = if is_markdown {
        extract_title_and_body(&content, path)
    } else {
        (filename_title(path), content)
    };

    Ok(IngestedDocument {
        title,
        body_markdown: body,
        images: vec![],
        source_path: Some(path.to_path_buf()),
        source_url: None,
        metadata: Default::default(),
    })
}

fn extract_title_and_body(content: &str, path: &Path) -> (String, String) {
    // Try to extract title from first H1
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(title) = trimmed.strip_prefix("# ") {
            return (title.trim().to_string(), content.to_string());
        }
        if !trimmed.is_empty() && !trimmed.starts_with("---") {
            break;
        }
    }
    (filename_title(path), content.to_string())
}

fn filename_title(path: &Path) -> String {
    path.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn ingest_markdown_with_title() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.md");
        std::fs::write(&path, "# My Title\n\nSome content here.").unwrap();
        let doc = ingest_text_file(&path).unwrap();
        assert_eq!(doc.title, "My Title");
        assert!(doc.body_markdown.contains("Some content"));
    }

    #[test]
    fn ingest_plain_text_uses_filename() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, "Just some text without headings.").unwrap();
        let doc = ingest_text_file(&path).unwrap();
        assert_eq!(doc.title, "notes");
    }
}

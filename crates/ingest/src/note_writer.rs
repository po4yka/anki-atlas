use std::path::{Path, PathBuf};

use crate::error::IngestError;
use crate::models::IngestedDocument;

/// Write an ingested document as an Obsidian-compatible markdown note.
///
/// Creates the file at `<vault_root>/<target_folder>/<slug>.md` with YAML
/// frontmatter and the document body. Images are written alongside the note.
pub fn write_obsidian_note(
    doc: &IngestedDocument,
    vault_root: &Path,
    target_folder: &str,
) -> Result<PathBuf, IngestError> {
    let slug = slugify(&doc.title);
    let folder = vault_root.join(target_folder);
    let note_path = folder.join(format!("{slug}.md"));

    std::fs::create_dir_all(&folder)?;

    let mut content = String::new();

    // YAML frontmatter
    content.push_str("---\n");
    content.push_str(&format!("title: \"{}\"\n", doc.title.replace('"', "'")));
    if let Some(url) = &doc.source_url {
        content.push_str(&format!("source: \"{url}\"\n"));
    }
    if let Some(path) = &doc.source_path {
        content.push_str(&format!("source_file: \"{}\"\n", path.display()));
    }
    for (key, value) in &doc.metadata {
        content.push_str(&format!("{key}: \"{}\"\n", value.replace('"', "'")));
    }
    content.push_str(&format!(
        "ingested_at: \"{}\"\n",
        chrono::Utc::now().to_rfc3339()
    ));
    content.push_str("---\n\n");

    // Title heading
    content.push_str(&format!("# {}\n\n", doc.title));

    // Body
    content.push_str(&doc.body_markdown);

    // Write images alongside the note
    for img in &doc.images {
        if !img.data.is_empty() {
            let img_path = folder.join(&img.filename);
            std::fs::write(&img_path, &img.data).map_err(|e| {
                IngestError::NoteWrite(format!("failed to write image {}: {e}", img.filename))
            })?;
        }
    }

    std::fs::write(&note_path, &content)?;

    Ok(note_path)
}

/// Convert a title to a filesystem-safe slug.
fn slugify(title: &str) -> String {
    title
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
        .chars()
        .take(80)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn slugify_handles_special_chars() {
        assert_eq!(slugify("Hello World!"), "hello-world");
        assert_eq!(
            slugify("Rust: A Systems Language"),
            "rust-a-systems-language"
        );
    }

    #[test]
    fn write_note_creates_file() {
        let dir = TempDir::new().unwrap();
        let doc = IngestedDocument {
            title: "Test Note".to_string(),
            body_markdown: "Some content here.".to_string(),
            images: vec![],
            source_url: Some("https://example.com".to_string()),
            source_path: None,
            metadata: Default::default(),
        };

        let path = write_obsidian_note(&doc, dir.path(), "ingested").unwrap();
        assert!(path.exists());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("title: \"Test Note\""));
        assert!(content.contains("source: \"https://example.com\""));
        assert!(content.contains("# Test Note"));
        assert!(content.contains("Some content here."));
    }
}

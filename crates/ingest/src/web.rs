use regex::Regex;
use std::sync::LazyLock;

use crate::error::IngestError;
use crate::models::{ExtractedImage, IngestedDocument};

static IMG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"<img[^>]+src="([^"]+)"[^>]*>"#).unwrap());

static TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());

/// Ingest a web page by fetching and extracting its article content.
pub async fn ingest_web_page(url: &str) -> Result<IngestedDocument, IngestError> {
    let response = reqwest::get(url).await?;
    let html = response.text().await?;

    let parsed_url =
        url::Url::parse(url).map_err(|e| IngestError::Web(format!("invalid URL: {e}")))?;
    let readable = readability::extractor::extract(&mut html.as_bytes(), &parsed_url)
        .map_err(|e| IngestError::Extraction(format!("readability extraction failed: {e}")))?;

    let title = if readable.title.is_empty() {
        url.to_string()
    } else {
        readable.title
    };

    let images = extract_image_refs(&readable.content, url);
    let body_markdown = html_to_markdown(&readable.content);

    Ok(IngestedDocument {
        title,
        body_markdown,
        images,
        source_url: Some(url.to_string()),
        source_path: None,
        metadata: Default::default(),
    })
}

/// Extract image references from HTML content.
fn extract_image_refs(html: &str, base_url: &str) -> Vec<ExtractedImage> {
    IMG_RE
        .captures_iter(html)
        .map(|cap| {
            let src = cap[1].to_string();
            let full_url = if src.starts_with("http") {
                src.clone()
            } else if src.starts_with("//") {
                format!("https:{src}")
            } else {
                format!(
                    "{}/{}",
                    base_url.trim_end_matches('/'),
                    src.trim_start_matches('/')
                )
            };
            let filename = full_url
                .rsplit('/')
                .next()
                .unwrap_or("image.png")
                .split('?')
                .next()
                .unwrap_or("image.png")
                .to_string();

            ExtractedImage {
                filename,
                data: Vec::new(),
                mime_type: "image/unknown".to_string(),
                caption: None,
            }
        })
        .collect()
}

/// Simple HTML to markdown conversion (strips tags, preserves text).
fn html_to_markdown(html: &str) -> String {
    let text = html
        .replace("<br>", "\n")
        .replace("<br/>", "\n")
        .replace("<br />", "\n")
        .replace("</p>", "\n\n")
        .replace("</h1>", "\n\n")
        .replace("</h2>", "\n\n")
        .replace("</h3>", "\n\n")
        .replace("</li>", "\n")
        .replace("<li>", "- ");

    TAG_RE.replace_all(&text, "").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_to_markdown_strips_tags() {
        let html = "<p>Hello <strong>world</strong></p><p>New paragraph</p>";
        let md = html_to_markdown(html);
        assert!(md.contains("Hello world"));
        assert!(md.contains("New paragraph"));
        assert!(!md.contains("<p>"));
    }

    #[test]
    fn extract_images_from_html() {
        let html = r#"<img src="https://example.com/photo.jpg" alt="test">"#;
        let images = extract_image_refs(html, "https://example.com");
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].filename, "photo.jpg");
    }
}

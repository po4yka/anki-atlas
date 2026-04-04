use std::path::{Path, PathBuf};

use generator::models::GeneratedCard;
use obsidian::parser::{ParsedNote, parse_note};
use obsidian::sync::{CardGenerator, GeneratedCardRef};
use serde::Serialize;

use crate::error::SurfaceError;

#[derive(Debug, Clone, Serialize)]
pub struct GeneratePreview {
    pub source_file: PathBuf,
    pub title: Option<String>,
    pub sections: Vec<String>,
    pub estimated_cards: usize,
    pub warnings: Vec<String>,
    pub cards: Vec<GeneratedCard>,
}

#[derive(Default)]
pub(super) struct PreviewCardGenerator;

impl CardGenerator for PreviewCardGenerator {
    fn generate(&self, note: &ParsedNote) -> Vec<GeneratedCardRef> {
        let sections: Vec<_> = note
            .sections
            .iter()
            .filter(|s| !s.heading.trim().is_empty() || !s.content.trim().is_empty())
            .collect();
        let count = if sections.is_empty() {
            1
        } else {
            sections.len()
        };

        (0..count)
            .map(|idx| GeneratedCardRef {
                slug: format!(
                    "{}-{}",
                    note.title
                        .as_deref()
                        .unwrap_or("note")
                        .to_lowercase()
                        .replace(' ', "-"),
                    idx + 1
                ),
                apf_html: note
                    .sections
                    .get(idx)
                    .map(|s| format!("<h2>{}</h2>\n<p>{}</p>", s.heading, s.content))
                    .unwrap_or_else(|| note.body.clone()),
            })
            .collect()
    }
}

pub struct GeneratePreviewService;

impl Default for GeneratePreviewService {
    fn default() -> Self {
        Self::new()
    }
}

impl GeneratePreviewService {
    pub fn new() -> Self {
        Self
    }

    pub fn preview(&self, file: &Path) -> Result<GeneratePreview, SurfaceError> {
        if !file.exists() {
            return Err(SurfaceError::PathNotFound(file.to_path_buf()));
        }
        let note = parse_note(file, file.parent())?;
        let estimated_cards = note
            .sections
            .iter()
            .filter(|s| !s.content.trim().is_empty())
            .count()
            .max(1);
        let warnings = if note.title.is_none() {
            vec!["No title detected; using filename in previews.".to_string()]
        } else {
            Vec::new()
        };
        let cards = note
            .sections
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.content.trim().is_empty())
            .map(|(idx, s)| GeneratedCard {
                card_index: (idx + 1) as u32,
                slug: format!(
                    "{}-{}",
                    note.title
                        .as_deref()
                        .unwrap_or("note")
                        .to_lowercase()
                        .replace(' ', "-"),
                    idx + 1
                ),
                lang: "unknown".to_string(),
                apf_html: format!("<h2>{}</h2>\n<p>{}</p>", s.heading, s.content),
                confidence: 0.5,
                content_hash: indexer::embeddings::content_hash("preview", &s.content),
                card_type: generator::CardType::default(),
            })
            .collect();

        Ok(GeneratePreview {
            source_file: file.to_path_buf(),
            title: note.title,
            sections: note.sections.into_iter().map(|s| s.heading).collect(),
            estimated_cards,
            warnings,
            cards,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use obsidian::parser::Section;
    use std::collections::HashMap;

    fn make_parsed_note(
        title: Option<&str>,
        sections: Vec<(&str, &str)>,
        body: &str,
    ) -> ParsedNote {
        ParsedNote {
            path: PathBuf::from("test.md"),
            frontmatter: HashMap::new(),
            content: String::new(),
            body: body.to_string(),
            sections: sections
                .into_iter()
                .map(|(heading, content)| Section {
                    heading: heading.to_string(),
                    content: content.to_string(),
                })
                .collect(),
            title: title.map(ToString::to_string),
        }
    }

    #[test]
    fn preview_card_generator_with_sections() {
        let generator = PreviewCardGenerator;
        let note = make_parsed_note(
            Some("Test Note"),
            vec![("Heading 1", "Content 1"), ("Heading 2", "Content 2")],
            "",
        );
        let cards = generator.generate(&note);
        assert_eq!(cards.len(), 2);
        assert!(cards[0].apf_html.contains("Heading 1"));
        assert!(cards[1].apf_html.contains("Content 2"));
    }

    #[test]
    fn preview_card_generator_no_sections_generates_one() {
        let generator = PreviewCardGenerator;
        let note = make_parsed_note(Some("Empty"), vec![], "Full body text");
        let cards = generator.generate(&note);
        assert_eq!(cards.len(), 1);
        assert_eq!(cards[0].apf_html, "Full body text");
    }

    #[test]
    fn preview_card_generator_slug_format() {
        let generator = PreviewCardGenerator;
        let note = make_parsed_note(Some("My Great Note"), vec![("S1", "C1"), ("S2", "C2")], "");
        let cards = generator.generate(&note);
        assert_eq!(cards[0].slug, "my-great-note-1");
        assert_eq!(cards[1].slug, "my-great-note-2");
    }

    #[test]
    fn generate_preview_service_default_does_not_panic() {
        let _service: GeneratePreviewService = Default::default();
    }
}

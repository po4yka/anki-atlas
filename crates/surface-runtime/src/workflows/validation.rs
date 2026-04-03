use std::path::{Path, PathBuf};

use serde::Serialize;
use validation::pipeline::{ValidationIssue, ValidationPipeline};
use validation::quality::QualityScore;
use validation::validators::{
    ContentValidator, FormatValidator, HtmlValidator, RelevanceValidator, TagValidator,
};

use crate::error::SurfaceError;

/// Controls whether quality checks are included in validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityCheck {
    Include,
    Skip,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidationSummary {
    pub source_file: PathBuf,
    pub is_valid: bool,
    pub issues: Vec<ValidationIssue>,
    pub quality: Option<QualityScore>,
}

pub struct ValidationService {
    pipeline: ValidationPipeline,
}

impl Default for ValidationService {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationService {
    pub fn new() -> Self {
        Self {
            pipeline: ValidationPipeline::new(vec![
                Box::new(ContentValidator::new()),
                Box::new(FormatValidator::new()),
                Box::new(HtmlValidator::new()),
                Box::new(TagValidator::new()),
                Box::new(RelevanceValidator::new()),
            ]),
        }
    }

    pub fn validate_file(
        &self,
        file: &Path,
        include_quality: QualityCheck,
    ) -> Result<ValidationSummary, SurfaceError> {
        if !file.exists() {
            return Err(SurfaceError::PathNotFound(file.to_path_buf()));
        }
        let content = std::fs::read_to_string(file)?;
        let (front, back, tags) = parse_validation_input(&content)?;
        let result = self.pipeline.run(&front, &back, &tags);
        let quality = (include_quality == QualityCheck::Include)
            .then(|| validation::quality::assess_quality_with_tags(&front, &back, &tags));

        Ok(ValidationSummary {
            source_file: file.to_path_buf(),
            is_valid: result.is_valid(),
            issues: result.issues,
            quality,
        })
    }
}

fn parse_validation_input(content: &str) -> Result<(String, String, Vec<String>), SurfaceError> {
    let mut parts = content.splitn(3, "\n---\n");
    let front = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            SurfaceError::InvalidInput(
                "validation input must contain front content before the first `---` separator"
                    .to_string(),
            )
        })?
        .to_string();
    let back = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            SurfaceError::InvalidInput(
                "validation input must contain back content after the first `---` separator"
                    .to_string(),
            )
        })?
        .to_string();
    let tags = parts
        .next()
        .map(|chunk| {
            chunk
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Ok((front, back, tags))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_validation_input_valid_three_parts() {
        let content = "What is Rust?\n---\nA systems language.\n---\ncs::rust\nprogramming";
        let (front, back, tags) = parse_validation_input(content).unwrap();
        assert_eq!(front, "What is Rust?");
        assert_eq!(back, "A systems language.");
        assert_eq!(tags, vec!["cs::rust", "programming"]);
    }

    #[test]
    fn parse_validation_input_two_parts_no_tags() {
        let content = "Front\n---\nBack";
        let (front, back, tags) = parse_validation_input(content).unwrap();
        assert_eq!(front, "Front");
        assert_eq!(back, "Back");
        assert!(tags.is_empty());
    }

    #[test]
    fn parse_validation_input_missing_back_returns_error() {
        let content = "Only front content";
        let result = parse_validation_input(content);
        assert!(matches!(result, Err(SurfaceError::InvalidInput(_))));
    }

    #[test]
    fn parse_validation_input_empty_front_returns_error() {
        let content = "\n---\nBack content";
        let result = parse_validation_input(content);
        assert!(matches!(result, Err(SurfaceError::InvalidInput(_))));
    }

    #[test]
    fn parse_validation_input_tags_trimmed_and_filtered() {
        let content = "Front\n---\nBack\n---\n  cs::basics  \n\n  rust  \n  ";
        let (_, _, tags) = parse_validation_input(content).unwrap();
        assert_eq!(tags, vec!["cs::basics", "rust"]);
    }

    #[test]
    fn validation_service_default_does_not_panic() {
        let _service = ValidationService::default();
    }
}

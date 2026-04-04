use serde::{Deserialize, Serialize};
use taxonomy::SkillRelevance;

/// Supported card types for generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CardType {
    /// Standard question/answer (front/back).
    #[default]
    Basic,
    /// Cloze deletion: `{{c1::answer}}` patterns in the text.
    Cloze,
    /// Multiple choice: question with labeled options (A/B/C/D).
    Mcq,
    /// Image-based card with occlusion regions.
    ImageOcclusion,
}

impl CardType {
    /// Map to Anki note type string.
    pub fn note_type(&self) -> &'static str {
        match self {
            Self::Basic => "APF::Simple",
            Self::Cloze => "APF::Cloze",
            Self::Mcq => "APF::Simple",
            Self::ImageOcclusion => "ImageOcclusion",
        }
    }
}

impl std::fmt::Display for CardType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Basic => write!(f, "basic"),
            Self::Cloze => write!(f, "cloze"),
            Self::Mcq => write!(f, "mcq"),
            Self::ImageOcclusion => write!(f, "image_occlusion"),
        }
    }
}

/// An image associated with a note for multimodal generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteImage {
    /// Relative path to the image file.
    pub path: String,
    /// MIME type (e.g., "image/png", "image/jpeg").
    pub mime_type: String,
    /// Alt text or filename description.
    pub description: Option<String>,
}

/// A single generated flashcard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCard {
    pub card_index: u32,
    pub slug: String,
    pub lang: String,
    pub apf_html: String,
    pub confidence: f32,
    pub content_hash: String,
    /// Card type (basic, cloze, mcq). Defaults to basic for backward compatibility.
    #[serde(default)]
    pub card_type: CardType,
}

/// Result of a card generation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub cards: Vec<GeneratedCard>,
    pub total_cards: usize,
    pub model_used: String,
    pub generation_time_secs: f64,
    pub warnings: Vec<String>,
}

/// Dependencies passed to generation agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationDeps {
    pub note_title: String,
    pub topic: String,
    pub language_tags: Vec<String>,
    pub source_file: String,
    /// Skill relevance bias for generation. When `Dead`, generation is skipped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skill_bias: Option<SkillRelevance>,
    /// Images associated with the source note for multimodal generation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<NoteImage>,
}

/// A single planned card from a split decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitPlan {
    pub card_number: u32,
    pub concept: String,
    pub question: String,
    pub answer_summary: String,
}

/// Result of a card splitting analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitDecision {
    pub should_split: bool,
    pub card_count: u32,
    pub plans: Vec<SplitPlan>,
    pub reasoning: String,
}

/// Severity level for validation issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Error,
    Warning,
}

/// A single validation issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub message: String,
    pub location: Option<String>,
}

/// Result of validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    /// Returns `true` if there are no errors (warnings are acceptable).
    pub fn is_valid(&self) -> bool {
        !self
            .issues
            .iter()
            .any(|i| matches!(i.severity, Severity::Error))
    }

    /// Returns only issues with `Severity::Error`.
    pub fn errors(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| matches!(i.severity, Severity::Error))
            .collect()
    }

    /// Returns only issues with `Severity::Warning`.
    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| matches!(i.severity, Severity::Warning))
            .collect()
    }
}

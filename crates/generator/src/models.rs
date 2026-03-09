use serde::{Deserialize, Serialize};

/// A single generated flashcard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCard {
    pub card_index: u32,
    pub slug: String,
    pub lang: String,
    pub apf_html: String,
    pub confidence: f32,
    pub content_hash: String,
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

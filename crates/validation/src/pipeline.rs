use serde::{Deserialize, Serialize};

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
    Info,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

/// A single validation issue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub message: String,
    /// Optional location hint (e.g. "front", "back", "tags").
    pub location: String,
}

impl ValidationIssue {
    pub fn error(message: impl Into<String>, location: impl Into<String>) -> Self {
        todo!()
    }

    pub fn warning(message: impl Into<String>, location: impl Into<String>) -> Self {
        todo!()
    }

    pub fn info(message: impl Into<String>, location: impl Into<String>) -> Self {
        todo!()
    }
}

/// Aggregated validation result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationResult {
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    /// True when no ERROR-level issues exist.
    pub fn is_valid(&self) -> bool {
        todo!()
    }

    /// Filter to ERROR-level issues only.
    pub fn errors(&self) -> Vec<&ValidationIssue> {
        todo!()
    }

    /// Filter to WARNING-level issues only.
    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        todo!()
    }

    /// Merge multiple results into one.
    pub fn merge(results: &[ValidationResult]) -> ValidationResult {
        todo!()
    }

    /// Empty (passing) result.
    pub fn ok() -> ValidationResult {
        todo!()
    }
}

/// Trait for validators. All implementations must be Send + Sync.
pub trait Validator: Send + Sync {
    fn validate(&self, front: &str, back: &str, tags: &[String]) -> ValidationResult;
}

/// Pipeline that chains validators in sequence and merges results.
pub struct ValidationPipeline {
    validators: Vec<Box<dyn Validator>>,
}

impl ValidationPipeline {
    pub fn new(validators: Vec<Box<dyn Validator>>) -> Self {
        todo!()
    }

    /// Run all validators and return merged result.
    pub fn run(&self, front: &str, back: &str, tags: &[String]) -> ValidationResult {
        todo!()
    }
}

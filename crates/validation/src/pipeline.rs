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
        match self {
            Severity::Error => write!(f, "error"),
            Severity::Warning => write!(f, "warning"),
            Severity::Info => write!(f, "info"),
        }
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
        Self {
            severity: Severity::Error,
            message: message.into(),
            location: location.into(),
        }
    }

    pub fn warning(message: impl Into<String>, location: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
            location: location.into(),
        }
    }

    pub fn info(message: impl Into<String>, location: impl Into<String>) -> Self {
        Self {
            severity: Severity::Info,
            message: message.into(),
            location: location.into(),
        }
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
        !self.issues.iter().any(|i| i.severity == Severity::Error)
    }

    /// Filter to ERROR-level issues only.
    pub fn errors(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Error)
            .collect()
    }

    /// Filter to WARNING-level issues only.
    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == Severity::Warning)
            .collect()
    }

    /// Merge multiple results into one.
    pub fn merge(results: &[ValidationResult]) -> ValidationResult {
        ValidationResult {
            issues: results
                .iter()
                .flat_map(|r| r.issues.clone())
                .collect(),
        }
    }

    /// Empty (passing) result.
    pub fn ok() -> ValidationResult {
        ValidationResult {
            issues: Vec::new(),
        }
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
        Self { validators }
    }

    /// Run all validators and return merged result.
    pub fn run(&self, front: &str, back: &str, tags: &[String]) -> ValidationResult {
        let results: Vec<ValidationResult> = self
            .validators
            .iter()
            .map(|v| v.validate(front, back, tags))
            .collect();
        ValidationResult::merge(&results)
    }
}

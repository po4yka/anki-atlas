use serde::{Deserialize, Serialize};

pub const MAX_LINE_WIDTH: usize = 88;
pub const MIN_TAGS: usize = 3;
pub const MAX_TAGS: usize = 6;

/// Result of APF linting/validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintResult {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl LintResult {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Validate APF card format against specification.
pub fn validate_apf(_apf_html: &str, _slug: Option<&str>) -> LintResult {
    todo!()
}

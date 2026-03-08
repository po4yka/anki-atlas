use crate::pipeline::{ValidationResult, Validator};

/// Check card content quality: empty fields, min/max length, unmatched code fences.
pub struct ContentValidator {
    pub min_length: usize,
    pub max_length: usize,
}

impl ContentValidator {
    pub fn new() -> Self {
        Self {
            min_length: 10,
            max_length: 5000,
        }
    }
}

impl Validator for ContentValidator {
    fn validate(&self, _front: &str, _back: &str, _tags: &[String]) -> ValidationResult {
        todo!()
    }
}

/// Check APF format compliance: trailing whitespace, consecutive blank lines.
pub struct FormatValidator;

impl FormatValidator {
    pub fn new() -> Self {
        Self
    }
}

impl Validator for FormatValidator {
    fn validate(&self, _front: &str, _back: &str, _tags: &[String]) -> ValidationResult {
        todo!()
    }
}

/// Validate HTML: balanced tags (stack-based), forbidden elements.
///
/// Forbidden tags: script, style, iframe, object, applet.
/// Void elements (br, img, hr, etc.) are skipped in balance checking.
pub struct HtmlValidator;

impl HtmlValidator {
    pub fn new() -> Self {
        Self
    }
}

impl Validator for HtmlValidator {
    fn validate(&self, _front: &str, _back: &str, _tags: &[String]) -> ValidationResult {
        todo!()
    }
}

/// Validate tags against conventions: no empty tags, no invalid chars,
/// max 20 tags, no duplicates.
///
/// Valid tag characters: [a-zA-Z0-9_:/-].
pub struct TagValidator {
    pub max_tags: usize,
}

impl TagValidator {
    pub fn new() -> Self {
        Self { max_tags: 20 }
    }
}

impl Validator for TagValidator {
    fn validate(&self, _front: &str, _back: &str, _tags: &[String]) -> ValidationResult {
        todo!()
    }
}

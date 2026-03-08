use std::collections::HashSet;

use crate::pipeline::{ValidationIssue, ValidationResult, Validator};

/// Check card content quality: empty fields, min/max length, unmatched code fences.
pub struct ContentValidator {
    pub min_length: usize,
    pub max_length: usize,
}

impl Default for ContentValidator {
    fn default() -> Self {
        Self {
            min_length: 10,
            max_length: 5000,
        }
    }
}

impl ContentValidator {
    pub fn new() -> Self {
        Self::default()
    }

    fn check_field(&self, content: &str, label: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        if content.trim().is_empty() {
            issues.push(ValidationIssue::error(
                format!("{label} side is empty"),
                label.to_lowercase(),
            ));
            return issues;
        }

        let len = content.len();
        if len < self.min_length {
            issues.push(ValidationIssue::warning(
                format!("{label} side is very short ({len} chars)"),
                label.to_lowercase(),
            ));
        }

        if len > self.max_length {
            issues.push(ValidationIssue::warning(
                format!("{label} side exceeds {} chars ({len})", self.max_length),
                label.to_lowercase(),
            ));
        }

        let fence_count = content.lines().filter(|l| l.trim_start().starts_with("```")).count();
        if fence_count % 2 != 0 {
            issues.push(ValidationIssue::error(
                format!("Unmatched code fence in {}", label.to_lowercase()),
                label.to_lowercase(),
            ));
        }

        issues
    }
}

impl Validator for ContentValidator {
    fn validate(&self, front: &str, back: &str, _tags: &[String]) -> ValidationResult {
        let mut issues = self.check_field(front, "Front");
        issues.extend(self.check_field(back, "Back"));
        ValidationResult { issues }
    }
}

/// Check APF format compliance: trailing whitespace, consecutive blank lines.
#[derive(Default)]
pub struct FormatValidator;

impl FormatValidator {
    pub fn new() -> Self {
        Self
    }

    fn check_field(&self, content: &str, label: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let location = label.to_lowercase();

        if content.lines().any(|line| line != line.trim_end()) {
            issues.push(ValidationIssue::warning(
                format!("Trailing whitespace in {location}"),
                &location,
            ));
        }

        if content.contains("\n\n\n") {
            issues.push(ValidationIssue::warning(
                format!("Consecutive blank lines in {location}"),
                &location,
            ));
        }

        issues
    }
}

impl Validator for FormatValidator {
    fn validate(&self, front: &str, back: &str, _tags: &[String]) -> ValidationResult {
        let mut issues = self.check_field(front, "Front");
        issues.extend(self.check_field(back, "Back"));
        ValidationResult { issues }
    }
}

/// Validate HTML: balanced tags (stack-based), forbidden elements.
///
/// Forbidden tags: script, style, iframe, object, applet.
/// Void elements (br, img, hr, etc.) are skipped in balance checking.
#[derive(Default)]
pub struct HtmlValidator;

const FORBIDDEN_TAGS: &[&str] = &["script", "style", "iframe", "object", "applet"];
const VOID_ELEMENTS: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
    "source", "track", "wbr",
];

impl HtmlValidator {
    pub fn new() -> Self {
        Self
    }

    fn check_field(&self, content: &str, label: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let location = label.to_lowercase();
        let mut stack: Vec<String> = Vec::new();
        let mut rest = content;

        while let Some(pos) = rest.find('<') {
            let after = &rest[pos + 1..];
            let Some(end) = after.find('>') else {
                break;
            };
            let tag_content = &after[..end];
            rest = &after[end + 1..];

            // Skip comments
            if tag_content.starts_with('!') {
                continue;
            }

            let is_closing = tag_content.starts_with('/');
            let tag_str = if is_closing {
                &tag_content[1..]
            } else {
                tag_content
            };

            // Extract tag name (first word, lowercase)
            let tag_name = tag_str
                .split(|c: char| c.is_whitespace() || c == '/')
                .next()
                .unwrap_or("")
                .to_lowercase();

            if tag_name.is_empty() {
                continue;
            }

            // Check forbidden
            if FORBIDDEN_TAGS.contains(&tag_name.as_str()) {
                issues.push(ValidationIssue::error(
                    format!("Forbidden HTML tag <{tag_name}> in {location}"),
                    &location,
                ));
            }

            // Skip void elements for balancing
            if VOID_ELEMENTS.contains(&tag_name.as_str()) {
                continue;
            }

            // Self-closing tags like <br/>
            if tag_content.ends_with('/') {
                continue;
            }

            if is_closing {
                if let Some(top) = stack.last() {
                    if *top == tag_name {
                        stack.pop();
                    } else {
                        issues.push(ValidationIssue::error(
                            format!(
                                "Mismatched closing tag </{tag_name}>, expected </{top}> in {location}"
                            ),
                            &location,
                        ));
                    }
                } else {
                    issues.push(ValidationIssue::error(
                        format!("Unexpected closing tag </{tag_name}> in {location}"),
                        &location,
                    ));
                }
            } else {
                stack.push(tag_name);
            }
        }

        for unclosed in &stack {
            issues.push(ValidationIssue::error(
                format!("Unclosed tag <{unclosed}> in {location}"),
                &location,
            ));
        }

        issues
    }
}

impl Validator for HtmlValidator {
    fn validate(&self, front: &str, back: &str, _tags: &[String]) -> ValidationResult {
        let mut issues = self.check_field(front, "Front");
        issues.extend(self.check_field(back, "Back"));
        ValidationResult { issues }
    }
}

/// Validate tags against conventions: no empty tags, no invalid chars,
/// max 20 tags, no duplicates.
///
/// Valid tag characters: [a-zA-Z0-9_:/-].
pub struct TagValidator {
    pub max_tags: usize,
}

impl Default for TagValidator {
    fn default() -> Self {
        Self { max_tags: 20 }
    }
}

impl TagValidator {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Validator for TagValidator {
    fn validate(&self, _front: &str, _back: &str, tags: &[String]) -> ValidationResult {
        let mut issues = Vec::new();

        for tag in tags {
            if tag.is_empty() {
                issues.push(ValidationIssue::error("Empty tag", "tags"));
                continue;
            }

            if !tag.chars().all(|c| c.is_ascii_alphanumeric() || "_:/-".contains(c)) {
                issues.push(ValidationIssue::warning(
                    format!("Tag '{tag}' contains invalid characters"),
                    "tags",
                ));
            }
        }

        if tags.len() > self.max_tags {
            issues.push(ValidationIssue::warning(
                format!("Too many tags ({}, max {})", tags.len(), self.max_tags),
                "tags",
            ));
        }

        let mut seen = HashSet::new();
        for tag in tags {
            if !seen.insert(tag) {
                issues.push(ValidationIssue::warning(
                    format!("Duplicate tag '{tag}'"),
                    "tags",
                ));
            }
        }

        ValidationResult { issues }
    }
}

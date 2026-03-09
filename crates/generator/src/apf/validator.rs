//! Validate HTML structure and Markdown formatting in APF cards.

use regex::Regex;
use std::sync::LazyLock;

/// Regex to match `<pre>...</pre>` blocks (non-greedy).
static PRE_BLOCK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<pre>.*?</pre>").unwrap());

/// Regex to match `<pre><code` (valid structure).
static PRE_CODE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<pre>\s*<code[\s>]").unwrap());

/// Regex to match standalone `<pre>` tags (for detecting pre-without-code).
static PRE_TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<pre>").unwrap());

/// Regex to match inline `<code>` not inside `<pre>`.
static INLINE_CODE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<code>").unwrap());

/// Regex to detect HTML tags in markdown content.
static HTML_TAG_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<[a-zA-Z][a-zA-Z0-9]*[^>]*>").unwrap());

/// Regex to match APF card header comments.
static CARD_HEADER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<!-- Card \d+").unwrap());

/// Result of Markdown validation.
#[derive(Debug, Clone)]
pub struct MarkdownValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Validate HTML structure used in APF cards.
/// Returns list of validation error messages (empty if valid).
pub fn validate_card_html(apf_html: &str) -> Vec<String> {
    if apf_html.is_empty() {
        return vec![];
    }

    let mut errors = Vec::new();

    // Check for backtick code fences (should be HTML, not markdown)
    if apf_html.contains("```") {
        errors.push("Backtick code fence detected; use <pre><code> in HTML".into());
    }

    // Strip <pre> blocks before checking for markdown markers
    let stripped = PRE_BLOCK_RE.replace_all(apf_html, "");

    // Check for markdown bold (**text**)
    if stripped.contains("**") {
        errors.push("Markdown bold markers (**) detected in HTML".into());
    }

    // Check for markdown italic (*text*) - single asterisk not part of **
    // Look for *word* pattern (single asterisk surrounding text)
    let no_bold = stripped.replace("**", "");
    if no_bold.contains('*') {
        errors.push("Markdown italic markers (*) detected in HTML".into());
    }

    // Check for <pre> without nested <code>
    for pre_match in PRE_TAG_RE.find_iter(apf_html) {
        let start = pre_match.start();
        // Check if <code> follows shortly after <pre>
        let rest = &apf_html[start..];
        if !PRE_CODE_RE.is_match(rest) {
            errors.push("<pre> without nested <code> tag detected".into());
        }
    }

    // Check for inline <code> outside <pre>
    let stripped_for_code = PRE_BLOCK_RE.replace_all(apf_html, "");
    if INLINE_CODE_RE.is_match(&stripped_for_code) {
        errors.push("Inline <code> outside <pre> detected".into());
    }

    errors
}

/// Validate Markdown structure (balanced fences, formatting markers).
pub fn validate_markdown(content: &str) -> MarkdownValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    if content.trim().is_empty() {
        return MarkdownValidationResult {
            is_valid: true,
            errors,
            warnings,
        };
    }

    // Check code fence balance
    let mut in_fence = false;
    let mut in_code_block = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            in_code_block = !in_code_block;
        }
        // Check for long lines inside code blocks
        if in_code_block && !trimmed.starts_with("```") && line.len() > 200 {
            warnings.push(format!(
                "Line exceeds 200 characters in code block ({} chars)",
                line.len()
            ));
        }
    }
    if in_fence {
        errors.push("Unclosed code fence detected".into());
    }

    // Collect content outside of balanced code fences for inline checks.
    // If a fence is unclosed, treat everything after the opening fence as outside.
    let mut outside_fence = String::new();
    let mut in_block = false;
    let mut fence_start_idx = 0;
    let mut segments: Vec<(bool, String)> = Vec::new();
    let mut current_segment = String::new();
    for line in content.lines() {
        if line.trim().starts_with("```") {
            segments.push((in_block, std::mem::take(&mut current_segment)));
            in_block = !in_block;
            if in_block {
                fence_start_idx = segments.len();
            }
            continue;
        }
        current_segment.push_str(line);
        current_segment.push('\n');
    }
    segments.push((in_block, current_segment));

    // If fence is still open (unbalanced), treat the fenced content as outside too
    for (i, (was_in_block, seg)) in segments.iter().enumerate() {
        if !was_in_block {
            outside_fence.push_str(seg);
        } else if in_fence && i >= fence_start_idx {
            // Unclosed fence: include this content for inline checks
            outside_fence.push_str(seg);
        }
    }
    let _ = fence_start_idx; // used above

    // Count inline backticks (single, not triple)
    let backtick_count = outside_fence.chars().filter(|&c| c == '`').count();
    if backtick_count % 2 != 0 {
        errors.push("Unbalanced backtick markers detected".into());
    }

    // Check for unbalanced bold markers (**)
    let bold_count = outside_fence.matches("**").count();
    if bold_count % 2 != 0 {
        errors.push("Unbalanced bold markers (**) detected".into());
    }

    // Check for HTML tags in markdown (warning)
    if HTML_TAG_RE.is_match(&outside_fence) {
        warnings.push("HTML tags detected in markdown content".into());
    }

    MarkdownValidationResult {
        is_valid: errors.is_empty(),
        errors,
        warnings,
    }
}

/// Validate APF document with Markdown content.
pub fn validate_apf_markdown(apf_content: &str) -> MarkdownValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    if apf_content.is_empty() {
        errors.push("Empty APF content".into());
        return MarkdownValidationResult {
            is_valid: false,
            errors,
            warnings,
        };
    }

    // Check required sentinels
    if !apf_content.contains("<!-- BEGIN_CARDS -->") {
        errors.push("Missing BEGIN_CARDS sentinel".into());
    }
    if !apf_content.contains("<!-- END_CARDS -->") {
        errors.push("Missing END_CARDS sentinel".into());
    }

    // Check for card header
    if !CARD_HEADER_RE.is_match(apf_content) {
        errors.push("No card header found (expected <!-- Card N ... -->)".into());
    }

    // Check for Title section
    if !apf_content.contains("<!-- Title -->") {
        errors.push("Missing Title section".into());
    }

    // Check for Key point sections (warning if missing)
    let has_key_point_code = apf_content.contains("<!-- Key point (code block");
    let has_key_point_notes = apf_content.contains("<!-- Key point notes -->");
    if !has_key_point_code && !has_key_point_notes {
        warnings.push("Missing Key point sections".into());
    }

    // Validate markdown content within sections
    // Extract text between sentinels and validate
    let section_content = extract_section_content(apf_content);
    if !section_content.is_empty() {
        let md_result = validate_markdown(&section_content);
        errors.extend(md_result.errors);
        warnings.extend(md_result.warnings);
    }

    MarkdownValidationResult {
        is_valid: errors.is_empty(),
        errors,
        warnings,
    }
}

/// Extract non-sentinel content from APF for markdown validation.
fn extract_section_content(apf_content: &str) -> String {
    let mut content = String::new();
    for line in apf_content.lines() {
        let trimmed = line.trim();
        // Skip sentinel/comment lines and manifest JSON
        if trimmed.starts_with("<!--") || trimmed.starts_with("{\"slug\"") {
            continue;
        }
        content.push_str(line);
        content.push('\n');
    }
    content
}

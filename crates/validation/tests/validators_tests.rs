use validation::{
    ContentValidator, FormatValidator, HtmlValidator, Severity, TagValidator, Validator,
};

// ─── ContentValidator ───────────────────────────────────────────────────────

#[test]
fn content_empty_front_is_error() {
    let v = ContentValidator::new();
    let result = v.validate("", "Valid back content here", &[]);
    assert!(!result.is_valid());
    let errors = result.errors();
    assert!(
        errors
            .iter()
            .any(|i| i.message.contains("Front side is empty"))
    );
}

#[test]
fn content_whitespace_only_front_is_error() {
    let v = ContentValidator::new();
    let result = v.validate("   \n\t  ", "Valid back content here", &[]);
    assert!(!result.is_valid());
    let errors = result.errors();
    assert!(
        errors
            .iter()
            .any(|i| i.message.contains("Front side is empty"))
    );
}

#[test]
fn content_empty_back_is_error() {
    let v = ContentValidator::new();
    let result = v.validate("Valid front content here", "", &[]);
    assert!(!result.is_valid());
    let errors = result.errors();
    assert!(
        errors
            .iter()
            .any(|i| i.message.contains("Back side is empty"))
    );
}

#[test]
fn content_short_front_is_warning() {
    let v = ContentValidator::new();
    let result = v.validate("Short", "Valid back content here", &[]);
    assert!(result.is_valid()); // warnings don't make invalid
    let warnings = result.warnings();
    assert!(
        warnings
            .iter()
            .any(|i| i.message.contains("Front side is very short"))
    );
}

#[test]
fn content_short_back_is_warning() {
    let v = ContentValidator::new();
    let result = v.validate("Valid front content here", "Short", &[]);
    assert!(result.is_valid());
    let warnings = result.warnings();
    assert!(
        warnings
            .iter()
            .any(|i| i.message.contains("Back side is very short"))
    );
}

#[test]
fn content_long_front_is_warning() {
    let v = ContentValidator::new();
    let long_front = "a".repeat(5001);
    let result = v.validate(&long_front, "Valid back content here", &[]);
    assert!(result.is_valid());
    let warnings = result.warnings();
    assert!(
        warnings
            .iter()
            .any(|i| i.message.contains("exceeds 5000 chars"))
    );
}

#[test]
fn content_long_back_is_warning() {
    let v = ContentValidator::new();
    let long_back = "a".repeat(5001);
    let result = v.validate("Valid front content here", &long_back, &[]);
    assert!(result.is_valid());
    let warnings = result.warnings();
    assert!(
        warnings
            .iter()
            .any(|i| i.message.contains("exceeds 5000 chars"))
    );
}

#[test]
fn content_unmatched_code_fence_front_is_error() {
    let v = ContentValidator::new();
    let front = "Some text\n```\ncode here\nmore text";
    let result = v.validate(front, "Valid back content here", &[]);
    assert!(!result.is_valid());
    let errors = result.errors();
    assert!(
        errors
            .iter()
            .any(|i| i.message.contains("Unmatched code fence"))
    );
}

#[test]
fn content_unmatched_code_fence_back_is_error() {
    let v = ContentValidator::new();
    let back = "Some text\n```\ncode here\nmore text";
    let result = v.validate("Valid front content here", back, &[]);
    assert!(!result.is_valid());
    let errors = result.errors();
    assert!(
        errors
            .iter()
            .any(|i| i.message.contains("Unmatched code fence"))
    );
}

#[test]
fn content_matched_code_fences_are_ok() {
    let v = ContentValidator::new();
    let front = "Question about:\n```\nsome code\n```\nWhat does it do?";
    let result = v.validate(front, "Valid back content here", &[]);
    // No code fence errors
    assert!(
        result
            .errors()
            .iter()
            .all(|i| !i.message.contains("code fence"))
    );
}

#[test]
fn content_valid_card_no_issues() {
    let v = ContentValidator::new();
    let result = v.validate(
        "What is the capital of France?",
        "The capital of France is Paris.",
        &[],
    );
    assert!(result.is_valid());
    assert!(result.issues.is_empty());
}

#[test]
fn content_both_empty_produces_two_errors() {
    let v = ContentValidator::new();
    let result = v.validate("", "", &[]);
    assert!(!result.is_valid());
    assert!(result.errors().len() >= 2);
}

#[test]
fn content_exactly_at_min_length_no_warning() {
    let v = ContentValidator::new();
    let front = "a".repeat(10);
    let result = v.validate(&front, "Valid back content here", &[]);
    // Front is exactly min_length, should not warn about being short
    assert!(
        result
            .warnings()
            .iter()
            .all(|i| !i.message.contains("Front side is very short"))
    );
}

#[test]
fn content_exactly_at_max_length_no_warning() {
    let v = ContentValidator::new();
    let front = "a".repeat(5000);
    let result = v.validate(&front, "Valid back content here", &[]);
    assert!(
        result
            .warnings()
            .iter()
            .all(|i| !i.message.contains("exceeds"))
    );
}

// ─── FormatValidator ────────────────────────────────────────────────────────

#[test]
fn format_trailing_whitespace_front_is_warning() {
    let v = FormatValidator::new();
    let result = v.validate("line with trailing space   \nnext line", "Valid back", &[]);
    assert!(result.is_valid());
    let warnings = result.warnings();
    assert!(!warnings.is_empty());
}

#[test]
fn format_trailing_whitespace_back_is_warning() {
    let v = FormatValidator::new();
    let result = v.validate("Valid front", "line with trailing space   \nnext line", &[]);
    assert!(result.is_valid());
    let warnings = result.warnings();
    assert!(!warnings.is_empty());
}

#[test]
fn format_consecutive_blank_lines_front_is_warning() {
    let v = FormatValidator::new();
    let result = v.validate("line1\n\n\nline2", "Valid back", &[]);
    assert!(result.is_valid());
    let warnings = result.warnings();
    assert!(!warnings.is_empty());
}

#[test]
fn format_consecutive_blank_lines_back_is_warning() {
    let v = FormatValidator::new();
    let result = v.validate("Valid front", "line1\n\n\nline2", &[]);
    assert!(result.is_valid());
    let warnings = result.warnings();
    assert!(!warnings.is_empty());
}

#[test]
fn format_clean_content_no_issues() {
    let v = FormatValidator::new();
    let result = v.validate("Clean front content", "Clean back content", &[]);
    assert!(result.is_valid());
    assert!(result.issues.is_empty());
}

#[test]
fn format_single_blank_line_is_ok() {
    let v = FormatValidator::new();
    let result = v.validate("line1\n\nline2", "Valid back", &[]);
    // Two newlines (one blank line) should be fine
    assert!(result.issues.is_empty());
}

// ─── HtmlValidator ──────────────────────────────────────────────────────────

#[test]
fn html_balanced_tags_no_issues() {
    let v = HtmlValidator::new();
    let result = v.validate("<div>content</div>", "Valid back", &[]);
    assert!(result.is_valid());
    assert!(result.issues.is_empty());
}

#[test]
fn html_unclosed_tag_is_error() {
    let v = HtmlValidator::new();
    let result = v.validate("<div>content", "Valid back", &[]);
    assert!(!result.is_valid());
}

#[test]
fn html_unexpected_closing_tag_is_error() {
    let v = HtmlValidator::new();
    let result = v.validate("content</span>", "Valid back", &[]);
    assert!(!result.is_valid());
}

#[test]
fn html_forbidden_script_is_error() {
    let v = HtmlValidator::new();
    let result = v.validate("<script>alert('xss')</script>", "Valid back", &[]);
    assert!(!result.is_valid());
    let errors = result.errors();
    assert!(errors.iter().any(|i| i.severity == Severity::Error));
}

#[test]
fn html_forbidden_style_is_error() {
    let v = HtmlValidator::new();
    let result = v.validate("<style>body{}</style>", "Valid back", &[]);
    assert!(!result.is_valid());
}

#[test]
fn html_forbidden_iframe_is_error() {
    let v = HtmlValidator::new();
    let result = v.validate("<iframe src='x'></iframe>", "Valid back", &[]);
    assert!(!result.is_valid());
}

#[test]
fn html_forbidden_object_is_error() {
    let v = HtmlValidator::new();
    let result = v.validate("<object></object>", "Valid back", &[]);
    assert!(!result.is_valid());
}

#[test]
fn html_forbidden_applet_is_error() {
    let v = HtmlValidator::new();
    let result = v.validate("<applet></applet>", "Valid back", &[]);
    assert!(!result.is_valid());
}

#[test]
fn html_void_elements_no_closing_needed() {
    let v = HtmlValidator::new();
    let result = v.validate("<br><img src='x'><hr>", "Valid back", &[]);
    assert!(result.is_valid());
    assert!(result.issues.is_empty());
}

#[test]
fn html_nested_tags_balanced() {
    let v = HtmlValidator::new();
    let result = v.validate("<div><p><b>text</b></p></div>", "Valid back", &[]);
    assert!(result.is_valid());
}

#[test]
fn html_mismatched_nesting_is_error() {
    let v = HtmlValidator::new();
    let result = v.validate("<div><p>text</div></p>", "Valid back", &[]);
    assert!(!result.is_valid());
}

#[test]
fn html_no_html_at_all_is_ok() {
    let v = HtmlValidator::new();
    let result = v.validate("plain text", "also plain", &[]);
    assert!(result.is_valid());
    assert!(result.issues.is_empty());
}

#[test]
fn html_checks_back_field_too() {
    let v = HtmlValidator::new();
    let result = v.validate("Valid front", "<div>unclosed", &[]);
    assert!(!result.is_valid());
}

#[test]
fn html_self_closing_void_in_back() {
    let v = HtmlValidator::new();
    let result = v.validate("Valid front", "text<br>more<img src='x'>end", &[]);
    assert!(result.is_valid());
}

// ─── TagValidator ───────────────────────────────────────────────────────────

#[test]
fn tag_empty_tag_is_error() {
    let v = TagValidator::new();
    let tags = vec!["valid".to_string(), "".to_string()];
    let result = v.validate("front", "back", &tags);
    assert!(!result.is_valid());
    let errors = result.errors();
    assert!(!errors.is_empty());
}

#[test]
fn tag_invalid_chars_is_warning() {
    let v = TagValidator::new();
    let tags = vec!["tag with spaces".to_string()];
    let result = v.validate("front", "back", &tags);
    assert!(result.is_valid()); // warning, not error
    let warnings = result.warnings();
    assert!(!warnings.is_empty());
}

#[test]
fn tag_valid_chars_no_issues() {
    let v = TagValidator::new();
    let tags = vec![
        "valid-tag".to_string(),
        "another_tag".to_string(),
        "scope:subtag".to_string(),
        "path/to/tag".to_string(),
        "MixedCase123".to_string(),
    ];
    let result = v.validate("front", "back", &tags);
    assert!(result.is_valid());
    assert!(result.issues.is_empty());
}

#[test]
fn tag_too_many_tags_is_warning() {
    let v = TagValidator::new();
    let tags: Vec<String> = (0..21).map(|i| format!("tag{i}")).collect();
    let result = v.validate("front", "back", &tags);
    assert!(result.is_valid());
    let warnings = result.warnings();
    assert!(!warnings.is_empty());
}

#[test]
fn tag_exactly_max_tags_no_warning() {
    let v = TagValidator::new();
    let tags: Vec<String> = (0..20).map(|i| format!("tag{i}")).collect();
    let result = v.validate("front", "back", &tags);
    assert!(result.warnings().iter().all(|i| !i.message.contains("20")));
}

#[test]
fn tag_duplicate_tags_is_warning() {
    let v = TagValidator::new();
    let tags = vec!["dup".to_string(), "other".to_string(), "dup".to_string()];
    let result = v.validate("front", "back", &tags);
    assert!(result.is_valid());
    let warnings = result.warnings();
    assert!(!warnings.is_empty());
}

#[test]
fn tag_empty_list_no_issues() {
    let v = TagValidator::new();
    let result = v.validate("front", "back", &[]);
    assert!(result.is_valid());
    assert!(result.issues.is_empty());
}

#[test]
fn tag_special_valid_chars() {
    let v = TagValidator::new();
    let tags = vec!["a-b_c:d/e".to_string()];
    let result = v.validate("front", "back", &tags);
    assert!(result.is_valid());
    assert!(result.issues.is_empty());
}

// ─── Send + Sync ────────────────────────────────────────────────────────────

common::assert_send_sync!(
    ContentValidator,
    FormatValidator,
    HtmlValidator,
    TagValidator
);

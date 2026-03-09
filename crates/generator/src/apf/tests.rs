use super::linter::*;
use super::renderer::*;

// ---------------------------------------------------------------------------
// Helper: build a minimal CardSpec for testing
// ---------------------------------------------------------------------------

fn minimal_spec() -> CardSpec {
    CardSpec {
        card_index: 1,
        slug: "test-card-01".into(),
        slug_base: None,
        lang: "en".into(),
        card_type: "Simple".into(),
        tags: vec!["python".into(), "testing".into(), "basics".into()],
        guid: "abc123".into(),
        source_path: None,
        source_anchor: None,
        title: "What is a variable?".into(),
        key_point_code: None,
        key_point_code_lang: None,
        key_point_notes: vec![],
        other_notes: None,
        extra: None,
    }
}

fn full_spec() -> CardSpec {
    CardSpec {
        card_index: 2,
        slug: "full-card-02".into(),
        slug_base: Some("full-card".into()),
        lang: "ru".into(),
        card_type: "Missing".into(),
        tags: vec!["rust".into(), "ownership".into(), "memory".into(), "borrow".into()],
        guid: "def456".into(),
        source_path: Some("notes/rust.md".into()),
        source_anchor: Some("ownership-section".into()),
        title: "Ownership in Rust".into(),
        key_point_code: Some("let x = String::from(\"hello\");\nlet y = x;".into()),
        key_point_code_lang: Some("rust".into()),
        key_point_notes: vec![
            "Values have a single owner".into(),
            "Assignment moves ownership".into(),
        ],
        other_notes: Some("See also: borrowing and references".into()),
        extra: Some("Chapter 4 of The Rust Programming Language".into()),
    }
}

// ---------------------------------------------------------------------------
// PROMPT_VERSION constant
// ---------------------------------------------------------------------------

#[test]
fn prompt_version_is_apf_v2_1() {
    assert_eq!(PROMPT_VERSION, "apf-v2.1");
}

// ---------------------------------------------------------------------------
// CardSpec serialization roundtrip
// ---------------------------------------------------------------------------

#[test]
fn card_spec_serialization_roundtrip() {
    let spec = full_spec();
    let json = serde_json::to_string(&spec).unwrap();
    let deserialized: CardSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.slug, spec.slug);
    assert_eq!(deserialized.card_index, spec.card_index);
    assert_eq!(deserialized.tags.len(), spec.tags.len());
}

// ---------------------------------------------------------------------------
// render() - sentinel markers
// ---------------------------------------------------------------------------

#[test]
fn render_contains_prompt_version_sentinel() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(
        html.contains("<!-- PROMPT_VERSION: apf-v2.1 -->"),
        "Missing PROMPT_VERSION sentinel in:\n{html}"
    );
}

#[test]
fn render_contains_begin_cards_sentinel() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(
        html.contains("<!-- BEGIN_CARDS -->"),
        "Missing BEGIN_CARDS sentinel"
    );
}

#[test]
fn render_contains_end_cards_sentinel() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(
        html.contains("<!-- END_CARDS -->"),
        "Missing END_CARDS sentinel"
    );
}

#[test]
fn render_ends_with_end_of_cards() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(
        html.trim_end().ends_with("END_OF_CARDS"),
        "Output must end with END_OF_CARDS, got:\n{html}"
    );
}

#[test]
fn render_contains_title_sentinel() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(html.contains("<!-- Title -->"), "Missing Title sentinel");
}

#[test]
fn render_contains_key_point_sentinel() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(
        html.contains("<!-- Key point"),
        "Missing Key point sentinel"
    );
}

#[test]
fn render_contains_manifest_sentinel() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(
        html.contains("<!-- manifest:"),
        "Missing manifest sentinel"
    );
}

// ---------------------------------------------------------------------------
// render() - card header
// ---------------------------------------------------------------------------

#[test]
fn render_card_header_format() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(
        html.contains(
            "<!-- Card 1 | slug: test-card-01 | CardType: Simple | Tags: python testing basics -->"
        ),
        "Card header format mismatch in:\n{html}"
    );
}

#[test]
fn render_card_header_with_empty_tags() {
    let mut spec = minimal_spec();
    spec.tags = vec![];
    let html = render(&spec);
    assert!(
        html.contains("| Tags:  -->"),
        "Empty tags should produce empty Tags field, got:\n{html}"
    );
}

// ---------------------------------------------------------------------------
// render() - title section
// ---------------------------------------------------------------------------

#[test]
fn render_includes_title_content() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(
        html.contains("<!-- Title -->\nWhat is a variable?"),
        "Title content missing"
    );
}

// ---------------------------------------------------------------------------
// render() - key point code block
// ---------------------------------------------------------------------------

#[test]
fn render_key_point_with_code() {
    let spec = full_spec();
    let html = render(&spec);
    assert!(
        html.contains("<pre><code class=\"language-rust\">"),
        "Code block should have language-rust class"
    );
}

#[test]
fn render_key_point_escapes_html_in_code() {
    let mut spec = minimal_spec();
    spec.key_point_code = Some("<script>alert('xss')</script>".into());
    spec.key_point_code_lang = Some("html".into());
    let html = render(&spec);
    assert!(
        !html.contains("<script>alert"),
        "HTML in code blocks must be escaped"
    );
    assert!(
        html.contains("&lt;script&gt;"),
        "HTML entities should be escaped"
    );
}

#[test]
fn render_key_point_without_code_lang_uses_plaintext() {
    let mut spec = minimal_spec();
    spec.key_point_code = Some("x = 1".into());
    spec.key_point_code_lang = None;
    let html = render(&spec);
    assert!(
        html.contains("language-plaintext"),
        "Missing language should default to plaintext"
    );
}

#[test]
fn render_key_point_no_code_renders_header_only() {
    let spec = minimal_spec();
    let html = render(&spec);
    // With no code, should still have the key point header comment
    assert!(html.contains("<!-- Key point (code block / image) -->"));
    // But no <pre><code> block
    assert!(
        !html.contains("<pre><code"),
        "No code block should be rendered when key_point_code is None"
    );
}

// ---------------------------------------------------------------------------
// render() - key point notes
// ---------------------------------------------------------------------------

#[test]
fn render_key_point_notes_as_list() {
    let spec = full_spec();
    let html = render(&spec);
    assert!(html.contains("<!-- Key point notes -->"));
    assert!(html.contains("<li>Values have a single owner</li>"));
    assert!(html.contains("<li>Assignment moves ownership</li>"));
    assert!(html.contains("<ul>"));
    assert!(html.contains("</ul>"));
}

#[test]
fn render_empty_key_point_notes_renders_empty_ul() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(
        html.contains("<ul></ul>"),
        "Empty notes should render as empty <ul>"
    );
}

// ---------------------------------------------------------------------------
// render() - other notes
// ---------------------------------------------------------------------------

#[test]
fn render_other_notes_present() {
    let spec = full_spec();
    let html = render(&spec);
    assert!(html.contains("<!-- Other notes -->"));
    assert!(html.contains("See also: borrowing and references"));
}

#[test]
fn render_other_notes_absent_renders_header_only() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(html.contains("<!-- Other notes -->"));
    // The line after "<!-- Other notes -->" should not have content
}

// ---------------------------------------------------------------------------
// render() - extra section
// ---------------------------------------------------------------------------

#[test]
fn render_extra_present() {
    let spec = full_spec();
    let html = render(&spec);
    assert!(html.contains("<!-- Extra -->"));
    assert!(html.contains("Chapter 4 of The Rust Programming Language"));
}

#[test]
fn render_extra_absent_renders_header_only() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(html.contains("<!-- Extra -->"));
}

// ---------------------------------------------------------------------------
// render() - manifest JSON
// ---------------------------------------------------------------------------

#[test]
fn render_manifest_contains_slug() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(
        html.contains("\"slug\":\"test-card-01\""),
        "Manifest must contain slug"
    );
}

#[test]
fn render_manifest_contains_lang_and_type() {
    let spec = full_spec();
    let html = render(&spec);
    assert!(html.contains("\"lang\":\"ru\""));
    assert!(html.contains("\"type\":\"Missing\""));
}

#[test]
fn render_manifest_contains_tags_array() {
    let spec = full_spec();
    let html = render(&spec);
    assert!(html.contains("\"tags\":[\"rust\",\"ownership\",\"memory\",\"borrow\"]"));
}

#[test]
fn render_manifest_includes_source_path_when_present() {
    let spec = full_spec();
    let html = render(&spec);
    assert!(html.contains("\"source_path\":\"notes/rust.md\""));
    assert!(html.contains("\"source_anchor\":\"ownership-section\""));
}

#[test]
fn render_manifest_excludes_source_when_absent() {
    let spec = minimal_spec();
    let html = render(&spec);
    assert!(
        !html.contains("source_path"),
        "source_path should be omitted when None"
    );
    assert!(
        !html.contains("source_anchor"),
        "source_anchor should be omitted when None"
    );
}

#[test]
fn render_manifest_slug_base_falls_back_to_slug_prefix() {
    let spec = minimal_spec();
    let html = render(&spec);
    // slug is "test-card-01", slug_base is None
    // Should fall back to slug.rsplit('-', 2)[0] -> "test-card"
    assert!(
        html.contains("\"slug_base\":\"test-card\""),
        "slug_base should fall back to slug prefix when None"
    );
}

#[test]
fn render_manifest_uses_explicit_slug_base() {
    let spec = full_spec();
    let html = render(&spec);
    assert!(html.contains("\"slug_base\":\"full-card\""));
}

#[test]
fn render_manifest_is_compact_json() {
    let spec = minimal_spec();
    let html = render(&spec);
    // Compact JSON means no spaces after : or ,
    // Find the manifest line
    let manifest_line = html.lines().find(|l| l.contains("<!-- manifest:")).unwrap();
    assert!(
        !manifest_line.contains("\": \""),
        "Manifest JSON should be compact (no spaces)"
    );
}

// ---------------------------------------------------------------------------
// render() - full output structure
// ---------------------------------------------------------------------------

#[test]
fn render_full_output_has_correct_sentinel_order() {
    let spec = full_spec();
    let html = render(&spec);

    let prompt_pos = html.find("<!-- PROMPT_VERSION:").unwrap();
    let begin_pos = html.find("<!-- BEGIN_CARDS -->").unwrap();
    let card_pos = html.find("<!-- Card ").unwrap();
    let title_pos = html.find("<!-- Title -->").unwrap();
    let key_point_pos = html.find("<!-- Key point").unwrap();
    let manifest_pos = html.find("<!-- manifest:").unwrap();
    let end_cards_pos = html.find("<!-- END_CARDS -->").unwrap();
    let end_of_cards_pos = html.find("END_OF_CARDS").unwrap();

    assert!(prompt_pos < begin_pos);
    assert!(begin_pos < card_pos);
    assert!(card_pos < title_pos);
    assert!(title_pos < key_point_pos);
    assert!(key_point_pos < manifest_pos);
    assert!(manifest_pos < end_cards_pos);
    assert!(end_cards_pos < end_of_cards_pos);
}

// ---------------------------------------------------------------------------
// render_batch()
// ---------------------------------------------------------------------------

#[test]
fn render_batch_empty_returns_empty_string() {
    let result = render_batch(&[]);
    assert_eq!(result, "");
}

#[test]
fn render_batch_single_card_same_as_render() {
    let spec = minimal_spec();
    let single = render(&spec);
    let batch = render_batch(&[spec]);
    assert_eq!(single, batch);
}

#[test]
fn render_batch_multiple_cards_separated_by_card_separator() {
    let spec1 = minimal_spec();
    let mut spec2 = full_spec();
    spec2.card_index = 2;

    let batch = render_batch(&[spec1, spec2]);
    let separator_count = batch.matches("<!-- CARD_SEPARATOR -->").count();
    assert_eq!(
        separator_count, 1,
        "Two cards should have exactly one separator"
    );
}

#[test]
fn render_batch_three_cards_has_two_separators() {
    let s1 = minimal_spec();
    let s2 = full_spec();
    let mut s3 = minimal_spec();
    s3.card_index = 3;
    s3.slug = "third-card-03".into();

    let batch = render_batch(&[s1, s2, s3]);
    let separator_count = batch.matches("<!-- CARD_SEPARATOR -->").count();
    assert_eq!(separator_count, 2);
}

#[test]
fn render_batch_each_card_has_own_sentinels() {
    let s1 = minimal_spec();
    let s2 = full_spec();
    let batch = render_batch(&[s1, s2]);

    let prompt_count = batch.matches("<!-- PROMPT_VERSION:").count();
    let begin_count = batch.matches("<!-- BEGIN_CARDS -->").count();
    let end_count = batch.matches("<!-- END_CARDS -->").count();
    let end_of_cards_count = batch.matches("END_OF_CARDS").count();

    assert_eq!(prompt_count, 2, "Each card should have its own PROMPT_VERSION");
    assert_eq!(begin_count, 2, "Each card should have its own BEGIN_CARDS");
    assert_eq!(end_count, 2, "Each card should have its own END_CARDS");
    assert_eq!(end_of_cards_count, 2, "Each card should have its own END_OF_CARDS");
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn render_special_characters_in_title() {
    let mut spec = minimal_spec();
    spec.title = "What does `fn main()` do?".into();
    let html = render(&spec);
    assert!(html.contains("What does `fn main()` do?"));
}

#[test]
fn render_unicode_content() {
    let mut spec = minimal_spec();
    spec.title = "Что такое переменная?".into();
    spec.lang = "ru".into();
    let html = render(&spec);
    assert!(html.contains("Что такое переменная?"));
}

#[test]
fn render_multiline_code() {
    let mut spec = minimal_spec();
    spec.key_point_code = Some("fn main() {\n    println!(\"Hello\");\n}".into());
    spec.key_point_code_lang = Some("rust".into());
    let html = render(&spec);
    assert!(html.contains("fn main() {\n    println!"));
}

#[test]
fn render_code_with_ampersand_and_quotes() {
    let mut spec = minimal_spec();
    spec.key_point_code = Some("x > 0 && y < 10 && z == \"hello\"".into());
    spec.key_point_code_lang = Some("python".into());
    let html = render(&spec);
    assert!(html.contains("&amp;"));
    assert!(html.contains("&gt;"));
    assert!(html.contains("&lt;"));
    assert!(html.contains("&quot;"));
}

// ---------------------------------------------------------------------------
// Send + Sync assertions
// ---------------------------------------------------------------------------

#[test]
fn card_spec_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CardSpec>();
}

// ===========================================================================
// APF Linter tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: generate valid APF HTML from a CardSpec via render()
// ---------------------------------------------------------------------------

fn valid_apf_html() -> String {
    render(&full_spec())
}

fn valid_apf_html_simple() -> String {
    render(&minimal_spec())
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#[test]
fn linter_max_line_width_is_88() {
    assert_eq!(MAX_LINE_WIDTH, 88);
}

#[test]
fn linter_min_tags_is_3() {
    assert_eq!(MIN_TAGS, 3);
}

#[test]
fn linter_max_tags_is_6() {
    assert_eq!(MAX_TAGS, 6);
}

// ---------------------------------------------------------------------------
// LintResult
// ---------------------------------------------------------------------------

#[test]
fn lint_result_is_valid_when_no_errors() {
    let result = LintResult {
        errors: vec![],
        warnings: vec!["some warning".into()],
    };
    assert!(result.is_valid());
}

#[test]
fn lint_result_is_invalid_when_errors() {
    let result = LintResult {
        errors: vec!["bad".into()],
        warnings: vec![],
    };
    assert!(!result.is_valid());
}

#[test]
fn lint_result_serialization_roundtrip() {
    let result = LintResult {
        errors: vec!["err1".into()],
        warnings: vec!["warn1".into()],
    };
    let json = serde_json::to_string(&result).unwrap();
    let deserialized: LintResult = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.errors, result.errors);
    assert_eq!(deserialized.warnings, result.warnings);
}

#[test]
fn lint_result_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<LintResult>();
}

// ---------------------------------------------------------------------------
// validate_apf() - valid input
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_valid_card_has_no_errors() {
    let html = valid_apf_html();
    let result = validate_apf(&html, None);
    assert!(
        result.is_valid(),
        "Valid APF should have no errors, got: {:?}",
        result.errors
    );
}

#[test]
fn validate_apf_valid_simple_card_has_no_errors() {
    let html = valid_apf_html_simple();
    let result = validate_apf(&html, None);
    assert!(
        result.is_valid(),
        "Valid simple APF should have no errors, got: {:?}",
        result.errors
    );
}

// ---------------------------------------------------------------------------
// validate_apf() - missing sentinels
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_missing_prompt_version_is_error() {
    let html = valid_apf_html().replace("<!-- PROMPT_VERSION: apf-v2.1 -->", "");
    let result = validate_apf(&html, None);
    assert!(!result.is_valid());
    assert!(
        result.errors.iter().any(|e| e.contains("PROMPT_VERSION")),
        "Should report missing PROMPT_VERSION, got: {:?}",
        result.errors
    );
}

#[test]
fn validate_apf_missing_begin_cards_is_error() {
    let html = valid_apf_html().replace("<!-- BEGIN_CARDS -->", "");
    let result = validate_apf(&html, None);
    assert!(!result.is_valid());
    assert!(
        result.errors.iter().any(|e| e.contains("BEGIN_CARDS")),
        "Should report missing BEGIN_CARDS, got: {:?}",
        result.errors
    );
}

#[test]
fn validate_apf_missing_end_cards_is_error() {
    let html = valid_apf_html().replace("<!-- END_CARDS -->", "");
    let result = validate_apf(&html, None);
    assert!(!result.is_valid());
    assert!(
        result.errors.iter().any(|e| e.contains("END_CARDS")),
        "Should report missing END_CARDS, got: {:?}",
        result.errors
    );
}

// ---------------------------------------------------------------------------
// validate_apf() - END_OF_CARDS final line
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_missing_end_of_cards_is_error() {
    let html = valid_apf_html().replace("END_OF_CARDS", "");
    let result = validate_apf(&html, None);
    assert!(!result.is_valid());
    assert!(
        result.errors.iter().any(|e| e.contains("END_OF_CARDS")),
        "Should report missing END_OF_CARDS, got: {:?}",
        result.errors
    );
}

// ---------------------------------------------------------------------------
// validate_apf() - card header format
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_invalid_card_header_format_is_error() {
    let html = valid_apf_html().replace(
        "<!-- Card 2 | slug: full-card-02 | CardType: Missing | Tags:",
        "<!-- Card 2 | slug: full-card-02 | type: Missing | Tags:",
    );
    let result = validate_apf(&html, None);
    assert!(!result.is_valid());
}

#[test]
fn validate_apf_comma_separated_tags_in_header_is_error() {
    // Replace space-separated tags with comma-separated
    let html = valid_apf_html().replace(
        "Tags: rust ownership memory borrow -->",
        "Tags: rust,ownership,memory,borrow -->",
    );
    let result = validate_apf(&html, None);
    assert!(!result.is_valid());
}

// ---------------------------------------------------------------------------
// validate_apf() - tag count validation
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_too_few_tags_is_error() {
    // Create a card with only 2 tags (below MIN_TAGS=3)
    let mut spec = full_spec();
    spec.tags = vec!["rust".into(), "basics".into()];
    let html = render(&spec);
    let result = validate_apf(&html, None);
    assert!(
        result.errors.iter().any(|e| e.contains("tag")),
        "Should report tag count violation, got: {:?}",
        result.errors
    );
}

#[test]
fn validate_apf_too_many_tags_is_error() {
    // Create a card with 7 tags (above MAX_TAGS=6)
    let mut spec = full_spec();
    spec.tags = vec![
        "rust".into(),
        "ownership".into(),
        "memory".into(),
        "borrow".into(),
        "lifetime".into(),
        "reference".into(),
        "extra".into(),
    ];
    let html = render(&spec);
    let result = validate_apf(&html, None);
    assert!(
        result.errors.iter().any(|e| e.contains("tag")),
        "Should report too many tags, got: {:?}",
        result.errors
    );
}

#[test]
fn validate_apf_exactly_min_tags_is_valid() {
    // 3 tags should be valid
    let html = valid_apf_html_simple(); // minimal_spec has 3 tags
    let result = validate_apf(&html, None);
    assert!(
        !result.errors.iter().any(|e| e.to_lowercase().contains("tag") && e.contains("3-6")),
        "3 tags should be valid, got errors: {:?}",
        result.errors
    );
}

#[test]
fn validate_apf_exactly_max_tags_is_valid() {
    let mut spec = full_spec();
    spec.tags = vec![
        "rust".into(),
        "ownership".into(),
        "memory".into(),
        "borrow".into(),
        "lifetime".into(),
        "reference".into(),
    ];
    let html = render(&spec);
    let result = validate_apf(&html, None);
    assert!(
        !result.errors.iter().any(|e| e.to_lowercase().contains("tag") && e.contains("3-6")),
        "6 tags should be valid, got errors: {:?}",
        result.errors
    );
}

// ---------------------------------------------------------------------------
// validate_apf() - tag format
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_uppercase_tag_is_warning() {
    let mut spec = full_spec();
    spec.tags = vec!["Rust".into(), "Ownership".into(), "memory".into(), "basics".into()];
    let html = render(&spec);
    let result = validate_apf(&html, None);
    assert!(
        result.warnings.iter().any(|w| w.contains("lowercase")),
        "Uppercase tags should produce warning, got: {:?}",
        result.warnings
    );
}

// ---------------------------------------------------------------------------
// validate_apf() - manifest validation
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_missing_manifest_is_error() {
    let html = valid_apf_html();
    // Remove the manifest line
    let html_no_manifest: String = html
        .lines()
        .filter(|l| !l.contains("<!-- manifest:"))
        .collect::<Vec<_>>()
        .join("\n");
    let result = validate_apf(&html_no_manifest, None);
    assert!(
        result.errors.iter().any(|e| e.contains("manifest")),
        "Should report missing manifest, got: {:?}",
        result.errors
    );
}

#[test]
fn validate_apf_invalid_manifest_json_is_error() {
    let html = valid_apf_html();
    // Replace manifest with invalid JSON
    let html_bad_manifest = html.replace(
        &html.lines().find(|l| l.contains("<!-- manifest:")).unwrap().to_string(),
        "<!-- manifest:{not valid json} -->",
    );
    let result = validate_apf(&html_bad_manifest, None);
    assert!(!result.is_valid());
    assert!(
        result.errors.iter().any(|e| e.contains("manifest") || e.contains("JSON")),
        "Should report invalid manifest JSON, got: {:?}",
        result.errors
    );
}

#[test]
fn validate_apf_manifest_slug_mismatch_is_error() {
    let html = valid_apf_html();
    // Change slug in manifest but not in header
    let html_mismatch = html.replace(
        "\"slug\":\"full-card-02\"",
        "\"slug\":\"wrong-slug-99\"",
    );
    let result = validate_apf(&html_mismatch, None);
    assert!(
        result.errors.iter().any(|e| e.contains("slug") && e.contains("mismatch")),
        "Should report manifest slug mismatch, got: {:?}",
        result.errors
    );
}

// ---------------------------------------------------------------------------
// validate_apf() - required field headers
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_missing_title_header_is_error() {
    let html = valid_apf_html().replace("<!-- Title -->", "");
    let result = validate_apf(&html, None);
    assert!(
        result.errors.iter().any(|e| e.contains("Title")),
        "Should report missing Title header, got: {:?}",
        result.errors
    );
}

#[test]
fn validate_apf_missing_key_point_header_is_error() {
    let html = valid_apf_html().replace("<!-- Key point (code block / image) -->", "");
    let result = validate_apf(&html, None);
    assert!(
        result.errors.iter().any(|e| e.contains("Key point")),
        "Should report missing Key point header, got: {:?}",
        result.errors
    );
}

#[test]
fn validate_apf_missing_key_point_notes_header_is_error() {
    let html = valid_apf_html().replace("<!-- Key point notes -->", "");
    let result = validate_apf(&html, None);
    assert!(
        result.errors.iter().any(|e| e.contains("Key point notes")),
        "Should report missing Key point notes header, got: {:?}",
        result.errors
    );
}

// ---------------------------------------------------------------------------
// validate_apf() - cloze density for Missing type
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_missing_type_no_cloze_is_warning() {
    // full_spec has card_type "Missing" but no cloze deletions
    let html = valid_apf_html();
    let result = validate_apf(&html, None);
    assert!(
        result.warnings.iter().any(|w| w.contains("cloze")),
        "Missing card without cloze should produce warning, got: {:?}",
        result.warnings
    );
}

#[test]
fn validate_apf_missing_type_non_dense_cloze_is_error() {
    // Insert cloze deletions with gap in numbering: {{c1::}} and {{c3::}} (missing c2)
    let mut spec = full_spec();
    spec.key_point_notes = vec![
        "{{c1::First cloze}}".into(),
        "{{c3::Third cloze}}".into(),
    ];
    let html = render(&spec);
    let result = validate_apf(&html, None);
    assert!(
        result.errors.iter().any(|e| e.contains("cloze") || e.contains("Cloze")),
        "Non-dense cloze should be error, got: {:?}",
        result.errors
    );
}

#[test]
fn validate_apf_missing_type_dense_cloze_no_error() {
    // Insert proper dense cloze: {{c1::}} and {{c2::}}
    let mut spec = full_spec();
    spec.key_point_notes = vec![
        "{{c1::First cloze}}".into(),
        "{{c2::Second cloze}}".into(),
    ];
    let html = render(&spec);
    let result = validate_apf(&html, None);
    assert!(
        !result.errors.iter().any(|e| e.to_lowercase().contains("cloze")),
        "Dense cloze should not produce error, got: {:?}",
        result.errors
    );
}

// ---------------------------------------------------------------------------
// validate_apf() - duplicate slug detection
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_duplicate_slugs_is_error() {
    let spec1 = minimal_spec();
    let mut spec2 = minimal_spec();
    spec2.card_index = 2;
    // Same slug as spec1: "test-card-01"

    let html = render_batch(&[spec1, spec2]);
    let result = validate_apf(&html, None);
    assert!(
        result.errors.iter().any(|e| e.to_lowercase().contains("duplicate")),
        "Duplicate slugs should be error, got: {:?}",
        result.errors
    );
}

#[test]
fn validate_apf_unique_slugs_no_error() {
    let spec1 = minimal_spec();
    let mut spec2 = minimal_spec();
    spec2.card_index = 2;
    spec2.slug = "another-card-02".into();

    let html = render_batch(&[spec1, spec2]);
    let result = validate_apf(&html, None);
    assert!(
        !result.errors.iter().any(|e| e.to_lowercase().contains("duplicate")),
        "Unique slugs should not produce duplicate error, got: {:?}",
        result.errors
    );
}

// ---------------------------------------------------------------------------
// validate_apf() - slug parameter matching
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_slug_mismatch_produces_warning() {
    let html = valid_apf_html(); // slug is "full-card-02"
    let result = validate_apf(&html, Some("expected-slug-01"));
    assert!(
        result.warnings.iter().any(|w| w.to_lowercase().contains("slug") && w.contains("mismatch")),
        "Slug mismatch with parameter should produce warning, got: {:?}",
        result.warnings
    );
}

#[test]
fn validate_apf_slug_matches_no_warning() {
    let html = valid_apf_html(); // slug is "full-card-02"
    let result = validate_apf(&html, Some("full-card-02"));
    assert!(
        !result.warnings.iter().any(|w| w.to_lowercase().contains("slug") && w.contains("mismatch")),
        "Matching slug should not produce mismatch warning, got: {:?}",
        result.warnings
    );
}

// ---------------------------------------------------------------------------
// validate_apf() - no card blocks
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_no_card_blocks_is_error() {
    let html = "<!-- PROMPT_VERSION: apf-v2.1 -->\n<!-- BEGIN_CARDS -->\n<!-- END_CARDS -->\nEND_OF_CARDS";
    let result = validate_apf(html, None);
    assert!(!result.is_valid());
    assert!(
        result.errors.iter().any(|e| e.contains("card") || e.contains("Card")),
        "No card blocks should be error, got: {:?}",
        result.errors
    );
}

// ---------------------------------------------------------------------------
// validate_apf() - empty input
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_empty_input_is_error() {
    let result = validate_apf("", None);
    assert!(!result.is_valid());
}

// ---------------------------------------------------------------------------
// validate_apf() - line width warnings
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_long_line_produces_warning() {
    let mut spec = full_spec();
    spec.other_notes = Some("x".repeat(100)); // > 88 chars
    let html = render(&spec);
    let result = validate_apf(&html, None);
    assert!(
        result.warnings.iter().any(|w| w.contains("88") || w.contains("character") || w.contains("width")),
        "Line > 88 chars should produce warning, got: {:?}",
        result.warnings
    );
}

// ---------------------------------------------------------------------------
// validate_apf() - manifest tags mismatch
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_manifest_tags_mismatch_is_warning() {
    let html = valid_apf_html();
    // Change tags in manifest only
    let html_modified = html.replace(
        "\"tags\":[\"rust\",\"ownership\",\"memory\",\"borrow\"]",
        "\"tags\":[\"rust\",\"ownership\"]",
    );
    let result = validate_apf(&html_modified, None);
    assert!(
        result.warnings.iter().any(|w| w.contains("tag") && w.contains("match")),
        "Manifest tags mismatch should produce warning, got: {:?}",
        result.warnings
    );
}

// ---------------------------------------------------------------------------
// validate_apf() - batch input with multiple cards
// ---------------------------------------------------------------------------

#[test]
fn validate_apf_batch_validates_all_cards() {
    let spec1 = minimal_spec();
    let spec2 = full_spec();
    let html = render_batch(&[spec1, spec2]);
    let result = validate_apf(&html, None);
    // Should validate both cards without crashing
    // (full_spec has Missing type without cloze, so at minimum a warning)
    assert!(result.errors.is_empty() || result.warnings.len() > 0);
}

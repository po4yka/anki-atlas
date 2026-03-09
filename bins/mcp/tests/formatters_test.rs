use anki_atlas_mcp::formatters::*;

// --- truncate ---

#[test]
fn truncate_short_text_unchanged() {
    let result = truncate("hello", 10);
    assert_eq!(result, "hello");
}

#[test]
fn truncate_exact_length_unchanged() {
    let result = truncate("hello", 5);
    assert_eq!(result, "hello");
}

#[test]
fn truncate_long_text_with_ellipsis() {
    let result = truncate("hello world", 8);
    assert_eq!(result, "hello...");
}

#[test]
fn truncate_replaces_newlines_with_space() {
    let result = truncate("line1\nline2\nline3", 50);
    assert_eq!(result, "line1 line2 line3");
}

#[test]
fn truncate_trims_whitespace() {
    let result = truncate("  spaced  ", 50);
    assert_eq!(result, "spaced");
}

#[test]
fn truncate_empty_string() {
    let result = truncate("", 10);
    assert_eq!(result, "");
}

// --- format_generate_result ---

#[test]
fn format_generate_result_with_title_and_sections() {
    let sections = vec![
        ("Introduction".to_string(), "Some intro content".to_string()),
        ("Details".to_string(), "Detail content here".to_string()),
    ];
    let result = format_generate_result(Some("My Note"), &sections, 500);

    assert!(result.contains("## Generation Preview"));
    assert!(result.contains("**Title**: My Note"));
    assert!(result.contains("**Body length**: 500 chars"));
    assert!(result.contains("**Sections**: 2"));
    assert!(result.contains("Introduction"));
    assert!(result.contains("Details"));
    assert!(result.contains("Estimated cards"));
}

#[test]
fn format_generate_result_no_title() {
    let sections = vec![];
    let result = format_generate_result(None, &sections, 0);

    assert!(result.contains("*(not detected)*"));
    assert!(result.contains("**Sections**: 0"));
}

#[test]
fn format_generate_result_estimated_cards_at_least_one() {
    let sections = vec![];
    let result = format_generate_result(Some("T"), &sections, 10);
    // Even with 0 sections, estimated cards should be at least 1
    assert!(result.contains("~1"));
}

// --- format_obsidian_sync_result ---

#[test]
fn format_obsidian_sync_result_with_notes() {
    let parsed = vec![
        ("note1.md".to_string(), Some("Note One".to_string()), 3),
        ("note2.md".to_string(), None, 0),
    ];
    let result = format_obsidian_sync_result(10, &parsed, "/my/vault");

    assert!(result.contains("## Obsidian Vault Scan"));
    assert!(result.contains("/my/vault"));
    assert!(result.contains("**Notes found**: 10"));
    assert!(result.contains("note1.md"));
    assert!(result.contains("Note One"));
    assert!(result.contains("*(untitled)*"));
}

#[test]
fn format_obsidian_sync_result_empty() {
    let result = format_obsidian_sync_result(0, &[], "/vault");
    assert!(result.contains("**Notes found**: 0"));
}

#[test]
fn format_obsidian_sync_result_truncates_at_20() {
    let parsed: Vec<(String, Option<String>, usize)> = (0..25)
        .map(|i| (format!("note{i}.md"), Some(format!("Title {i}")), 1))
        .collect();
    let result = format_obsidian_sync_result(25, &parsed, "/vault");
    // Should mention remaining notes
    assert!(result.contains("more notes"));
}

// --- format_tag_audit_result ---

#[test]
fn format_tag_audit_result_all_valid() {
    let results = vec![
        ("cs::algo".to_string(), vec![], None, vec![]),
        ("math::calc".to_string(), vec![], None, vec![]),
    ];
    let output = format_tag_audit_result(&results);

    assert!(output.contains("## Tag Audit Results"));
    assert!(output.contains("**Total tags**: 2"));
    assert!(output.contains("**Valid**: 2"));
    assert!(output.contains("**With issues**: 0"));
    assert!(output.contains("All tags are valid"));
}

#[test]
fn format_tag_audit_result_with_issues() {
    let results = vec![
        (
            "Bad_Tag".to_string(),
            vec!["not kebab-case".to_string()],
            Some("bad-tag".to_string()),
            vec!["bad-tag".to_string()],
        ),
        ("good-tag".to_string(), vec![], None, vec![]),
    ];
    let output = format_tag_audit_result(&results);

    assert!(output.contains("**Valid**: 1"));
    assert!(output.contains("**With issues**: 1"));
    assert!(output.contains("Bad_Tag"));
    assert!(output.contains("not kebab-case"));
}

#[test]
fn format_tag_audit_result_empty() {
    let results: Vec<(String, Vec<String>, Option<String>, Vec<String>)> = vec![];
    let output = format_tag_audit_result(&results);
    assert!(output.contains("**Total tags**: 0"));
    assert!(output.contains("All tags are valid"));
}

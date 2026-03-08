use std::collections::HashMap;

use super::{parse_frontmatter, write_frontmatter};

// ── parse_frontmatter: valid YAML ──────────────────────────────────────

#[test]
fn parse_extracts_simple_key_value_pairs() {
    let content = "---\ntitle: Hello\ntags: world\n---\nBody text";
    let fm = parse_frontmatter(content).unwrap();
    assert_eq!(fm.get("title").and_then(|v| v.as_str()), Some("Hello"));
    assert_eq!(fm.get("tags").and_then(|v| v.as_str()), Some("world"));
}

#[test]
fn parse_extracts_nested_yaml() {
    let content = "---\nmeta:\n  author: Alice\n  version: 2\n---\nBody";
    let fm = parse_frontmatter(content).unwrap();
    let meta = fm.get("meta").unwrap().as_mapping().unwrap();
    let author = meta
        .get(serde_yaml::Value::String("author".into()))
        .and_then(|v| v.as_str());
    assert_eq!(author, Some("Alice"));
}

#[test]
fn parse_extracts_list_values() {
    let content = "---\ntags:\n  - rust\n  - tdd\n---\nBody";
    let fm = parse_frontmatter(content).unwrap();
    let tags = fm.get("tags").unwrap().as_sequence().unwrap();
    assert_eq!(tags.len(), 2);
    assert_eq!(tags[0].as_str(), Some("rust"));
    assert_eq!(tags[1].as_str(), Some("tdd"));
}

// ── parse_frontmatter: no frontmatter ──────────────────────────────────

#[test]
fn parse_returns_empty_map_when_no_frontmatter() {
    let content = "Just a regular markdown note\nwith no frontmatter.";
    let fm = parse_frontmatter(content).unwrap();
    assert!(fm.is_empty());
}

#[test]
fn parse_returns_empty_map_for_empty_string() {
    let fm = parse_frontmatter("").unwrap();
    assert!(fm.is_empty());
}

#[test]
fn parse_returns_empty_map_for_single_delimiter() {
    let content = "---\nNot closed frontmatter";
    let fm = parse_frontmatter(content).unwrap();
    assert!(fm.is_empty());
}

// ── parse_frontmatter: backtick preprocessing ──────────────────────────

#[test]
fn parse_strips_backticks_from_values() {
    let content = "---\ntitle: `Hello World`\ndate: `2024-01-01`\n---\nBody";
    let fm = parse_frontmatter(content).unwrap();
    assert_eq!(fm.get("title").and_then(|v| v.as_str()), Some("Hello World"));
    assert_eq!(
        fm.get("date").and_then(|v| v.as_str()),
        Some("2024-01-01")
    );
}

#[test]
fn parse_backticks_only_in_frontmatter_not_body() {
    let content = "---\ntitle: `Test`\n---\nBody with `code` blocks";
    let fm = parse_frontmatter(content).unwrap();
    assert_eq!(fm.get("title").and_then(|v| v.as_str()), Some("Test"));
}

// ── parse_frontmatter: malformed YAML ──────────────────────────────────

#[test]
fn parse_returns_error_on_malformed_yaml() {
    let content = "---\ntitle: :\n  invalid: [yaml\n---\nBody";
    let result = parse_frontmatter(content);
    assert!(result.is_err());
}

// ── write_frontmatter: replace existing ────────────────────────────────

#[test]
fn write_replaces_existing_frontmatter() {
    let content = "---\ntitle: Old\n---\nBody text here";
    let mut data = HashMap::new();
    data.insert(
        "title".to_string(),
        serde_yaml::Value::String("New".into()),
    );
    let result = write_frontmatter(&data, content).unwrap();
    assert!(result.contains("title"));
    assert!(result.contains("New"));
    assert!(!result.contains("Old"));
    assert!(result.contains("Body text here"));
    assert!(result.starts_with("---\n"));
}

// ── write_frontmatter: add to content without frontmatter ──────────────

#[test]
fn write_adds_frontmatter_to_content_without_one() {
    let content = "Just a body with no frontmatter.";
    let mut data = HashMap::new();
    data.insert(
        "title".to_string(),
        serde_yaml::Value::String("Added".into()),
    );
    let result = write_frontmatter(&data, content).unwrap();
    assert!(result.starts_with("---\n"));
    assert!(result.contains("title"));
    assert!(result.contains("Added"));
    assert!(result.contains("Just a body with no frontmatter."));
}

#[test]
fn write_preserves_body_content() {
    let content = "---\nold: value\n---\nLine 1\nLine 2\nLine 3";
    let mut data = HashMap::new();
    data.insert(
        "new_key".to_string(),
        serde_yaml::Value::String("new_val".into()),
    );
    let result = write_frontmatter(&data, content).unwrap();
    assert!(result.contains("Line 1"));
    assert!(result.contains("Line 2"));
    assert!(result.contains("Line 3"));
}

#[test]
fn write_empty_data_produces_empty_frontmatter() {
    let content = "Body only";
    let data = HashMap::new();
    let result = write_frontmatter(&data, content).unwrap();
    assert!(result.starts_with("---\n"));
    assert!(result.contains("Body only"));
}

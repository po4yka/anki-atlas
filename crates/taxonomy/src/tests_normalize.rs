use crate::normalize::*;

// === normalize_tag ===

#[test]
fn normalize_tag_kotlin_coroutines() {
    assert_eq!(normalize_tag("Kotlin::Coroutines"), "kotlin_coroutines");
}

#[test]
fn normalize_tag_meta_preserved() {
    assert_eq!(normalize_tag("difficulty::hard"), "difficulty::hard");
}

#[test]
fn normalize_tag_unknown_kebab() {
    assert_eq!(normalize_tag("Unknown_Tag"), "unknown-tag");
}

#[test]
fn normalize_tag_empty() {
    assert_eq!(normalize_tag(""), "");
}

#[test]
fn normalize_tag_whitespace() {
    assert_eq!(normalize_tag("  "), "");
}

#[test]
fn normalize_tag_trim() {
    assert_eq!(normalize_tag("  coroutines  "), "kotlin_coroutines");
}

#[test]
fn normalize_tag_meta_atomic() {
    assert_eq!(normalize_tag("atomic"), "atomic");
}

#[test]
fn normalize_tag_meta_lang() {
    assert_eq!(normalize_tag("lang::en"), "lang::en");
}

#[test]
fn normalize_tag_meta_source() {
    assert_eq!(normalize_tag("source::anki"), "source::anki");
}

#[test]
fn normalize_tag_meta_context() {
    assert_eq!(normalize_tag("context::review"), "context::review");
}

#[test]
fn normalize_tag_already_topic_prefix() {
    // Tags starting with topic prefix pass through
    assert_eq!(normalize_tag("kotlin_coroutines"), "kotlin_coroutines");
    assert_eq!(normalize_tag("android_compose"), "android_compose");
    assert_eq!(normalize_tag("cs_algorithms"), "cs_algorithms");
    assert_eq!(normalize_tag("bias_self"), "bias_self");
}

#[test]
fn normalize_tag_cognitive_bias_special_case() {
    assert_eq!(normalize_tag("cognitive_bias"), "cognitive_bias");
}

#[test]
fn normalize_tag_known_mapping() {
    assert_eq!(normalize_tag("flow"), "kotlin_flow");
    assert_eq!(normalize_tag("algorithms"), "cs_algorithms");
}

#[test]
fn normalize_tag_unknown_with_slashes() {
    assert_eq!(normalize_tag("foo/bar"), "foo-bar");
}

#[test]
fn normalize_tag_unknown_with_double_colon() {
    assert_eq!(normalize_tag("Unknown::Thing"), "unknown-thing");
}

#[test]
fn normalize_tag_collapse_hyphens() {
    assert_eq!(normalize_tag("foo__bar"), "foo-bar");
}

#[test]
fn normalize_tag_strip_leading_trailing_hyphens() {
    // After transformation, leading/trailing hyphens should be stripped
    assert_eq!(normalize_tag("_foo_"), "foo");
}

// === normalize_tags ===

#[test]
fn normalize_tags_deduplicates() {
    let result = normalize_tags(&["coroutines", "Kotlin::Coroutines"]);
    assert_eq!(result, vec!["kotlin_coroutines"]);
}

#[test]
fn normalize_tags_sorts() {
    let result = normalize_tags(&["flow", "coroutines"]);
    assert_eq!(result, vec!["kotlin_coroutines", "kotlin_flow"]);
}

#[test]
fn normalize_tags_removes_empties() {
    let result = normalize_tags(&["", "  ", "coroutines"]);
    assert_eq!(result, vec!["kotlin_coroutines"]);
}

#[test]
fn normalize_tags_empty_input() {
    let result = normalize_tags(&[]);
    assert!(result.is_empty());
}

// === validate_tag ===

#[test]
fn validate_tag_empty() {
    let issues = validate_tag("");
    assert!(!issues.is_empty());
    assert!(issues[0].contains("empty"));
}

#[test]
fn validate_tag_whitespace_only() {
    let issues = validate_tag("   ");
    assert!(!issues.is_empty());
    assert!(issues[0].contains("empty") || issues[0].contains("whitespace"));
}

#[test]
fn validate_tag_kotlin_foobar_underscore() {
    // kotlin_foobar is NOT a known topic tag, so should warn about underscore
    let issues = validate_tag("kotlin_foobar");
    assert!(!issues.is_empty());
    assert!(
        issues.iter().any(|i| i.contains("::") && i.contains("_")),
        "expected underscore warning, got: {issues:?}"
    );
}

#[test]
fn validate_tag_known_topic_tag_no_issues() {
    // kotlin_coroutines IS a known topic tag, so no underscore warning
    let issues = validate_tag("kotlin_coroutines");
    assert!(issues.is_empty(), "expected no issues, got: {issues:?}");
}

#[test]
fn validate_tag_too_deep() {
    let issues = validate_tag("a::b::c");
    assert!(!issues.is_empty());
    assert!(
        issues.iter().any(|i| i.contains("deep") || i.contains("levels")),
        "expected depth warning, got: {issues:?}"
    );
}

#[test]
fn validate_tag_slash_separator() {
    let issues = validate_tag("foo/bar");
    assert!(!issues.is_empty());
    assert!(
        issues.iter().any(|i| i.contains("/")),
        "expected slash warning, got: {issues:?}"
    );
}

#[test]
fn validate_tag_uppercase_prefix() {
    let issues = validate_tag("Kotlin::Coroutines");
    assert!(
        issues.iter().any(|i| i.contains("lowercase")),
        "expected uppercase prefix warning, got: {issues:?}"
    );
}

#[test]
fn validate_tag_duplicate_colons() {
    let issues = validate_tag("a::::b");
    assert!(
        issues.iter().any(|i| i.contains("Duplicate")),
        "expected duplicate separator warning, got: {issues:?}"
    );
}

#[test]
fn validate_tag_duplicate_hyphens() {
    let issues = validate_tag("foo--bar");
    assert!(
        issues.iter().any(|i| i.contains("Duplicate") || i.contains("'-'")),
        "expected duplicate hyphen warning, got: {issues:?}"
    );
}

#[test]
fn validate_tag_valid_prefixed_tag() {
    let issues = validate_tag("kotlin::basics");
    assert!(issues.is_empty(), "expected no issues, got: {issues:?}");
}

#[test]
fn validate_tag_cognitive_bias_special() {
    let issues = validate_tag("cognitive_bias");
    assert!(issues.is_empty(), "expected no issues for cognitive_bias, got: {issues:?}");
}

// === suggest_tag ===

#[test]
fn suggest_tag_coroutine_finds_coroutines() {
    let suggestions = suggest_tag("coroutine", 5);
    assert!(
        suggestions.iter().any(|s| s == "coroutines"),
        "expected 'coroutines' in suggestions, got: {suggestions:?}"
    );
}

#[test]
fn suggest_tag_empty_input() {
    let suggestions = suggest_tag("", 5);
    assert!(suggestions.is_empty());
}

#[test]
fn suggest_tag_respects_max_results() {
    let suggestions = suggest_tag("a", 3);
    assert!(suggestions.len() <= 3);
}

#[test]
fn suggest_tag_exact_match_excluded() {
    let suggestions = suggest_tag("coroutines", 10);
    // Exact match should NOT appear in suggestions
    assert!(
        !suggestions.iter().any(|s| s.to_lowercase() == "coroutines"),
        "exact match should be excluded, got: {suggestions:?}"
    );
}

// === is_meta_tag ===

#[test]
fn is_meta_tag_difficulty_easy() {
    assert!(is_meta_tag("difficulty::easy"));
}

#[test]
fn is_meta_tag_lang_ru() {
    assert!(is_meta_tag("lang::ru"));
}

#[test]
fn is_meta_tag_atomic() {
    assert!(is_meta_tag("atomic"));
}

#[test]
fn is_meta_tag_source() {
    assert!(is_meta_tag("source::book"));
}

#[test]
fn is_meta_tag_context() {
    assert!(is_meta_tag("context::review"));
}

#[test]
fn is_meta_tag_not_topic() {
    assert!(!is_meta_tag("kotlin_coroutines"));
}

#[test]
fn is_meta_tag_not_random() {
    assert!(!is_meta_tag("foobar"));
}

// === is_topic_tag ===

#[test]
fn is_topic_tag_cs_algorithms() {
    assert!(is_topic_tag("cs_algorithms"));
}

#[test]
fn is_topic_tag_kotlin_coroutines() {
    assert!(is_topic_tag("kotlin_coroutines"));
}

#[test]
fn is_topic_tag_android_compose() {
    assert!(is_topic_tag("android_compose"));
}

#[test]
fn is_topic_tag_bias_self() {
    assert!(is_topic_tag("bias_self"));
}

#[test]
fn is_topic_tag_cognitive_bias_special() {
    assert!(is_topic_tag("cognitive_bias"));
}

#[test]
fn is_topic_tag_unknown_prefix() {
    // Tags starting with a topic prefix are considered topic tags
    // even if not in the canonical set
    assert!(is_topic_tag("kotlin_unknown_thing"));
}

#[test]
fn is_topic_tag_not_meta() {
    assert!(!is_topic_tag("difficulty::easy"));
}

#[test]
fn is_topic_tag_not_random() {
    assert!(!is_topic_tag("foobar"));
}

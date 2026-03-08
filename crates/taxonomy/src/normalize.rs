/// Normalize a single tag to canonical form.
pub fn normalize_tag(_tag: &str) -> String {
    todo!()
}

/// Normalize a list of tags: normalize each, deduplicate, sort, remove empties.
pub fn normalize_tags(_tags: &[&str]) -> Vec<String> {
    todo!()
}

/// Validate a tag and return a list of issues (empty if valid).
pub fn validate_tag(_tag: &str) -> Vec<String> {
    todo!()
}

/// Suggest up to `max_results` close matches from the known taxonomy.
pub fn suggest_tag(_input_tag: &str, _max_results: usize) -> Vec<String> {
    todo!()
}

/// Check if a tag is a meta tag.
pub fn is_meta_tag(_tag: &str) -> bool {
    false
}

/// Check if a tag is a recognized topic tag.
pub fn is_topic_tag(_tag: &str) -> bool {
    false
}

use crate::tags::{is_known_topic_tag, lookup_tag, META_TAGS, META_TAG_PREFIXES, TOPIC_PREFIXES};

/// Normalize a single tag to canonical form.
pub fn normalize_tag(tag: &str) -> String {
    let tag = tag.trim();
    if tag.is_empty() {
        return String::new();
    }

    // Keep meta tags as-is
    if META_TAGS.contains(&tag) || META_TAG_PREFIXES.iter().any(|p| tag.starts_with(p)) {
        return tag.to_owned();
    }

    // Keep already-prefixed topic tags as-is
    if TOPIC_PREFIXES.iter().any(|p| tag.starts_with(p)) || tag == "cognitive_bias" {
        return tag.to_owned();
    }

    // Map known tags
    if let Some(mapped) = lookup_tag(tag) {
        return mapped.to_owned();
    }

    // Unknown tag: lowercase, kebab-case
    let mut normalized = tag
        .to_lowercase()
        .replace('_', "-")
        .replace("::", "-")
        .replace('/', "-");

    // Collapse multiple hyphens
    while normalized.contains("--") {
        normalized = normalized.replace("--", "-");
    }

    normalized.trim_matches('-').to_owned()
}

/// Normalize a list of tags: normalize each, deduplicate, sort, remove empties.
pub fn normalize_tags(tags: &[&str]) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    for tag in tags {
        let result = normalize_tag(tag);
        if !result.is_empty() {
            seen.insert(result);
        }
    }
    seen.into_iter().collect()
}

/// Validate a tag and return a list of issues (empty if valid).
pub fn validate_tag(tag: &str) -> Vec<String> {
    let mut issues = Vec::new();

    if tag.is_empty() || tag.trim().is_empty() {
        issues.push("Tag is empty or whitespace-only".to_owned());
        return issues;
    }

    let tag = tag.trim();

    // Check for underscore as prefix separator (anti-pattern)
    for prefix in ["kotlin_", "android_", "cs_", "bias_"] {
        if tag.starts_with(prefix) && tag != "cognitive_bias" && !is_known_topic_tag(tag) {
            let suffix = &tag[prefix.len()..];
            let prefix_name = &prefix[..prefix.len() - 1];
            issues.push(format!(
                "Use '::' for domain prefix, not '_': '{tag}' -> '{prefix_name}::{suffix}'"
            ));
            break;
        }
    }

    // Check for slash as hierarchy separator
    if tag.contains('/') {
        issues.push(format!("Use '::' for hierarchy, not '/': '{tag}'"));
    }

    // Check hierarchy depth
    let parts: Vec<&str> = tag.split("::").collect();
    if parts.len() > 2 {
        issues.push(format!(
            "Tag too deep (max 2 levels): '{tag}' has {} levels",
            parts.len()
        ));
    }

    // Check for uppercase prefix
    if tag.contains("::") {
        let prefix_part = parts[0];
        if prefix_part != prefix_part.to_lowercase() {
            issues.push(format!("Prefix should be lowercase: '{prefix_part}'"));
        }
        if parts.len() > 1 {
            let topic_part = parts[1];
            if topic_part != topic_part.to_lowercase()
                && !topic_part
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_uppercase())
            {
                issues.push(format!("Topic should be lowercase: '{topic_part}'"));
            }
        }
    }

    // Check for underscores as word separators (within tag parts)
    for part in &parts {
        if part.contains('_') && !is_known_topic_tag(tag) && tag != "cognitive_bias" {
            issues.push(format!("Use '-' between words, not '_': '{part}'"));
            break;
        }
    }

    // Check for duplicate separators
    if tag.contains("::::") {
        issues.push("Duplicate '::' separator found".to_owned());
    }
    if tag.contains("--") {
        issues.push("Duplicate '-' separator found".to_owned());
    }

    issues
}

/// Suggest up to `max_results` close matches from the known taxonomy.
pub fn suggest_tag(input_tag: &str, max_results: usize) -> Vec<String> {
    let input_tag = input_tag.trim();
    if input_tag.is_empty() {
        return Vec::new();
    }

    let tag_lower = input_tag.to_lowercase();
    let max_distance: usize = 2;

    // Collect candidates from TAG_MAP keys + values + all topic tags
    let mut candidates = std::collections::HashSet::new();

    // Add all keys and values from the tag map
    // We need to iterate via the static map
    for (key, value) in &crate::tags::TAG_MAP {
        candidates.insert(*key);
        candidates.insert(*value);
    }

    // Add all topic tags
    for set in crate::tags::ALL_TOPIC_TAG_SETS {
        for tag in *set {
            candidates.insert(*tag);
        }
    }

    let mut scored: Vec<(usize, &str)> = Vec::new();

    for candidate in &candidates {
        let candidate_lower = candidate.to_lowercase();

        // Skip exact match
        if candidate_lower == tag_lower {
            continue;
        }

        // Prefix match gets score 0 (best)
        if candidate_lower.starts_with(&tag_lower) || tag_lower.starts_with(&candidate_lower) {
            scored.push((0, candidate));
            continue;
        }

        // Quick length check
        let len_diff = if input_tag.len() > candidate.len() {
            input_tag.len() - candidate.len()
        } else {
            candidate.len() - input_tag.len()
        };
        if len_diff > max_distance {
            continue;
        }

        // Simple character difference count
        let differences: usize = tag_lower
            .chars()
            .zip(candidate_lower.chars())
            .filter(|(a, b)| a != b)
            .count()
            + len_diff;

        if differences <= max_distance {
            scored.push((differences, candidate));
        }
    }

    scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored
        .into_iter()
        .take(max_results)
        .map(|(_, s)| s.to_owned())
        .collect()
}

/// Check if a tag is a meta tag.
pub fn is_meta_tag(tag: &str) -> bool {
    META_TAGS.contains(&tag) || META_TAG_PREFIXES.iter().any(|p| tag.starts_with(p))
}

/// Check if a tag is a recognized topic tag.
pub fn is_topic_tag(tag: &str) -> bool {
    is_known_topic_tag(tag)
        || TOPIC_PREFIXES.iter().any(|p| tag.starts_with(p))
        || tag == "cognitive_bias"
}

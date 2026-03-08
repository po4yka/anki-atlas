use std::collections::HashMap;

use crate::error::ObsidianError;

/// Extract YAML frontmatter from note content.
/// Returns empty map if no frontmatter block is found.
pub fn parse_frontmatter(
    content: &str,
) -> Result<HashMap<String, serde_yaml::Value>, ObsidianError> {
    let Some(yaml_str) = extract_raw_frontmatter(content) else {
        return Ok(HashMap::new());
    };

    // Preprocess: strip backticks from values
    let preprocessed = preprocess_yaml(yaml_str);

    let map: HashMap<String, serde_yaml::Value> =
        serde_yaml::from_str(&preprocessed).map_err(|e| ObsidianError::Yaml(e.to_string()))?;

    Ok(map)
}

/// Write or replace YAML frontmatter in note content.
pub fn write_frontmatter(
    data: &HashMap<String, serde_yaml::Value>,
    content: &str,
) -> Result<String, ObsidianError> {
    let yaml_str =
        serde_yaml::to_string(data).map_err(|e| ObsidianError::Yaml(e.to_string()))?;

    let body = extract_body(content);

    let mut result = String::from("---\n");
    // serde_yaml adds a trailing newline; only append non-empty yaml
    if !data.is_empty() {
        result.push_str(&yaml_str);
    }
    result.push_str("---\n");
    result.push_str(body);

    Ok(result)
}

/// Extract raw YAML string between opening and closing `---` delimiters.
/// Returns `None` if no valid frontmatter block is found.
fn extract_raw_frontmatter(content: &str) -> Option<&str> {
    let content = content.strip_prefix("---\n")?;
    let end = content.find("\n---")?;
    Some(&content[..end])
}

/// Get body content (everything after frontmatter, or the full content if none).
fn extract_body(content: &str) -> &str {
    if let Some(rest) = content.strip_prefix("---\n") {
        if let Some(end) = rest.find("\n---") {
            let after_delim = &rest[end + 4..]; // skip "\n---"
            return after_delim.strip_prefix('\n').unwrap_or(after_delim);
        }
    }
    content
}

/// Strip backticks wrapping YAML values: `value` -> value
fn preprocess_yaml(yaml: &str) -> String {
    yaml.lines()
        .map(|line| {
            if let Some(colon_pos) = line.find(": ") {
                let (key, val) = line.split_at(colon_pos + 2);
                let trimmed = val.trim();
                if trimmed.starts_with('`') && trimmed.ends_with('`') && trimmed.len() >= 2 {
                    let stripped = &trimmed[1..trimmed.len() - 1];
                    return format!("{key}{stripped}");
                }
            }
            line.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests;

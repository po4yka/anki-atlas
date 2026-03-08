use std::collections::HashMap;

use crate::error::ObsidianError;

/// Extract YAML frontmatter from note content.
/// Returns empty map if no frontmatter block is found.
pub fn parse_frontmatter(
    content: &str,
) -> Result<HashMap<String, serde_yaml::Value>, ObsidianError> {
    let (Some(yaml_str), _body) = split_frontmatter(content) else {
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
    let yaml_str = serde_yaml::to_string(data).map_err(|e| ObsidianError::Yaml(e.to_string()))?;

    let (_yaml, body) = split_frontmatter(content);

    let mut result = String::from("---\n");
    // serde_yaml adds a trailing newline; only append non-empty yaml
    if !data.is_empty() {
        result.push_str(&yaml_str);
    }
    result.push_str("---\n");
    result.push_str(body);

    Ok(result)
}

/// Split content into optional raw YAML frontmatter and body.
/// Returns `(None, full_content)` if no valid frontmatter block is found.
pub(crate) fn split_frontmatter(content: &str) -> (Option<&str>, &str) {
    let Some(rest) = content.strip_prefix("---\n") else {
        return (None, content);
    };
    let Some(end) = rest.find("\n---") else {
        return (None, content);
    };
    let yaml = &rest[..end];
    let after_delim = &rest[end + 4..]; // skip "\n---"
    let body = after_delim.strip_prefix('\n').unwrap_or(after_delim);
    (Some(yaml), body)
}

/// Strip backticks wrapping YAML values: `value` -> value
fn preprocess_yaml(yaml: &str) -> String {
    yaml.lines()
        .map(|line| {
            if let Some(colon_pos) = line.find(": ") {
                let (key, val) = line.split_at(colon_pos + 2);
                if let Some(stripped) = val
                    .trim()
                    .strip_prefix('`')
                    .and_then(|s| s.strip_suffix('`'))
                {
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

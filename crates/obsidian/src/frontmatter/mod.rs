use std::collections::HashMap;

use crate::error::ObsidianError;

/// Extract YAML frontmatter from note content.
/// Returns empty map if no frontmatter block is found.
pub fn parse_frontmatter(
    _content: &str,
) -> Result<HashMap<String, serde_yaml::Value>, ObsidianError> {
    todo!()
}

/// Write or replace YAML frontmatter in note content.
pub fn write_frontmatter(
    _data: &HashMap<String, serde_yaml::Value>,
    _content: &str,
) -> Result<String, ObsidianError> {
    todo!()
}

#[cfg(test)]
mod tests;

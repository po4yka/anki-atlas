use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

/// Regex for card header: `<!-- Card N | slug: X | CardType: Y | Tags: Z -->`
static CARD_HEADER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^<!-- Card \d+ \| slug: ([^ |]+) \| CardType: ([^ |]+) \| Tags: (.*?) -->$")
        .expect("valid card header regex")
});

/// Regex for cloze deletion markers: `{{cN::`
static CLOZE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{\{c(\d+)::").expect("valid cloze regex"));

pub const MAX_LINE_WIDTH: usize = 88;
pub const MIN_TAGS: usize = 3;
pub const MAX_TAGS: usize = 6;

/// Result of APF linting/validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintResult {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl LintResult {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Parsed card block extracted from APF HTML.
struct CardBlock {
    slug: String,
    card_type: String,
    header_tags: Vec<String>,
    has_title: bool,
    has_key_point: bool,
    has_key_point_notes: bool,
    manifest_json: Option<String>,
    content: String,
}

/// Parse a card header line into (slug, card_type, tags).
fn parse_card_header(line: &str) -> Option<(String, String, Vec<String>)> {
    let caps = CARD_HEADER_RE.captures(line)?;
    let slug = caps[1].to_string();
    let card_type = caps[2].to_string();
    let tags_str = caps[3].trim();
    let tags: Vec<String> = if tags_str.is_empty() {
        vec![]
    } else {
        tags_str.split(' ').map(|s| s.to_string()).collect()
    };
    Some((slug, card_type, tags))
}

/// Validate APF card format against specification.
pub fn validate_apf(apf_html: &str, slug: Option<&str>) -> LintResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Check required sentinels
    if !apf_html.contains("<!-- PROMPT_VERSION: apf-v2.1 -->") {
        errors.push("Missing PROMPT_VERSION sentinel".to_string());
    }
    if !apf_html.contains("<!-- BEGIN_CARDS -->") {
        errors.push("Missing BEGIN_CARDS sentinel".to_string());
    }
    if !apf_html.contains("<!-- END_CARDS -->") {
        errors.push("Missing END_CARDS sentinel".to_string());
    }
    if !apf_html.contains("END_OF_CARDS") {
        errors.push("Missing END_OF_CARDS sentinel".to_string());
    }

    // Line width warnings
    for (i, line) in apf_html.lines().enumerate() {
        if line.len() > MAX_LINE_WIDTH && !line.starts_with("<!-- manifest:") {
            warnings.push(format!(
                "Line {} exceeds {} character width limit",
                i + 1,
                MAX_LINE_WIDTH
            ));
        }
    }

    // Split into card blocks by finding card headers
    let lines: Vec<&str> = apf_html.lines().collect();
    let mut card_blocks: Vec<CardBlock> = Vec::new();

    let mut i = 0;
    while i < lines.len() {
        if let Some((card_slug, card_type, header_tags)) = parse_card_header(lines[i]) {
            // Collect lines until next card header or END_CARDS
            let start = i + 1;
            let mut end = start;
            while end < lines.len()
                && parse_card_header(lines[end]).is_none()
                && !lines[end].contains("<!-- END_CARDS -->")
                && !lines[end].contains("<!-- CARD_SEPARATOR -->")
            {
                end += 1;
            }
            let content = lines[start..end].join("\n");

            let has_title = content.contains("<!-- Title -->");
            let has_key_point = content.contains("<!-- Key point (code block");
            let has_key_point_notes = content.contains("<!-- Key point notes -->");

            let manifest_json = content.lines().find_map(|l| {
                l.strip_prefix("<!-- manifest:")
                    .and_then(|rest| rest.strip_suffix(" -->"))
                    .map(|json| json.to_string())
            });

            card_blocks.push(CardBlock {
                slug: card_slug,
                card_type,
                header_tags,
                has_title,
                has_key_point,
                has_key_point_notes,
                manifest_json,
                content,
            });
            i = end;
        } else {
            i += 1;
        }
    }

    // Must have at least one card block
    if card_blocks.is_empty() {
        errors.push("No Card blocks found".to_string());
        return LintResult { errors, warnings };
    }

    // Duplicate slug detection
    let mut seen_slugs = HashSet::new();
    for block in &card_blocks {
        if !seen_slugs.insert(&block.slug) {
            errors.push(format!("Duplicate slug: {}", block.slug));
        }
    }

    // Validate each card block
    for block in &card_blocks {
        // Check for comma-separated tags in header
        if block
            .header_tags
            .len()
            == 1
            && block.header_tags[0].contains(',')
        {
            errors.push(format!(
                "Card {}: tags must be space-separated, not comma-separated",
                block.slug
            ));
        }

        // Tag count validation
        let tag_count = block.header_tags.len();
        if !(MIN_TAGS..=MAX_TAGS).contains(&tag_count) {
            errors.push(format!(
                "Card {}: expected {}-{} tags, found {}",
                block.slug, MIN_TAGS, MAX_TAGS, tag_count
            ));
        }

        // Tag format: warn on uppercase
        if block.header_tags.iter().any(|t| t != &t.to_lowercase()) {
            warnings.push(format!(
                "Card {}: tags should be lowercase",
                block.slug
            ));
        }

        // Required field headers
        if !block.has_title {
            errors.push(format!("Card {}: missing Title header", block.slug));
        }
        if !block.has_key_point {
            errors.push(format!("Card {}: missing Key point header", block.slug));
        }
        if !block.has_key_point_notes {
            errors.push(format!(
                "Card {}: missing Key point notes header",
                block.slug
            ));
        }

        // Manifest validation
        match &block.manifest_json {
            None => {
                errors.push(format!("Card {}: missing manifest", block.slug));
            }
            Some(json_str) => {
                match serde_json::from_str::<serde_json::Value>(json_str) {
                    Err(_) => {
                        errors.push(format!(
                            "Card {}: invalid manifest JSON",
                            block.slug
                        ));
                    }
                    Ok(manifest) => {
                        // Slug mismatch between header and manifest
                        if let Some(manifest_slug) = manifest.get("slug").and_then(|v| v.as_str())
                        {
                            if manifest_slug != block.slug {
                                errors.push(format!(
                                    "Card {}: manifest slug mismatch (header: {}, manifest: {})",
                                    block.slug, block.slug, manifest_slug
                                ));
                            }
                        }

                        // Tags mismatch between header and manifest
                        if let Some(manifest_tags) = manifest.get("tags").and_then(|v| v.as_array())
                        {
                            let m_tags: Vec<&str> = manifest_tags
                                .iter()
                                .filter_map(|v| v.as_str())
                                .collect();
                            let h_tags: Vec<&str> =
                                block.header_tags.iter().map(|s| s.as_str()).collect();
                            if m_tags != h_tags {
                                warnings.push(format!(
                                    "Card {}: manifest tags don't match header tags",
                                    block.slug
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Cloze validation for Missing type
        if block.card_type == "Missing" {
            let cloze_numbers: Vec<u32> = CLOZE_RE
                .captures_iter(&block.content)
                .filter_map(|c| c[1].parse().ok())
                .collect();

            if cloze_numbers.is_empty() {
                warnings.push(format!(
                    "Card {}: Missing type has no cloze deletions",
                    block.slug
                ));
            } else {
                // Check dense numbering: 1..=max with no gaps
                let max_n = *cloze_numbers.iter().max().unwrap_or(&0);
                let unique: HashSet<u32> = cloze_numbers.into_iter().collect();
                let is_dense = (1..=max_n).all(|n| unique.contains(&n));
                if !is_dense {
                    errors.push(format!(
                        "Card {}: Cloze numbering is not dense (gaps detected)",
                        block.slug
                    ));
                }
            }
        }

        // Slug parameter matching
        if let Some(expected_slug) = slug {
            if block.slug != expected_slug {
                warnings.push(format!(
                    "Card slug mismatch: expected {}, found {}",
                    expected_slug, block.slug
                ));
            }
        }
    }

    LintResult { errors, warnings }
}

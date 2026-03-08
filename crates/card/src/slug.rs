use sha2::{Digest, Sha256};
use unicode_normalization::UnicodeNormalization;

/// Maximum length for a single slug component (topic or keyword).
pub const MAX_COMPONENT_LENGTH: usize = 50;
/// Maximum total slug length.
pub const MAX_SLUG_LENGTH: usize = 100;

/// Extracted components from a slug.
#[derive(Debug, Clone, Default)]
pub struct SlugComponents {
    pub topic: String,
    pub keyword: String,
    pub index: u32,
    pub lang: String,
}

/// Slug generation errors.
#[derive(Debug, thiserror::Error)]
pub enum SlugError {
    #[error("invalid hash length: {0} (must be 1..=64)")]
    InvalidHashLength(usize),
    #[error("invalid index: must be non-negative")]
    InvalidIndex,
    #[error("invalid lang: must be exactly 2 characters, got '{0}'")]
    InvalidLang(String),
    #[error("empty input: {field}")]
    EmptyInput { field: &'static str },
}

/// Collapse runs of multiple hyphens into a single hyphen.
fn collapse_hyphens(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_hyphen = false;
    for ch in s.chars() {
        if ch == '-' {
            if !prev_hyphen {
                out.push('-');
            }
            prev_hyphen = true;
        } else {
            out.push(ch);
            prev_hyphen = false;
        }
    }
    out
}

/// Validate a language code: must be exactly 2 ASCII letters. Returns lowercased.
fn validate_lang(lang: &str) -> Result<String, SlugError> {
    let lower = lang.to_lowercase();
    if lower.len() != 2 || !lower.chars().all(|c| c.is_ascii_alphabetic()) {
        return Err(SlugError::InvalidLang(lang.to_string()));
    }
    Ok(lower)
}

/// Slugify topic and keyword with defaults for empty inputs.
fn slugify_components(topic: &str, keyword: &str) -> (String, String) {
    let topic_slug = if topic.is_empty() {
        "untitled".to_string()
    } else {
        SlugService::slugify(topic)
    };
    let keyword_slug = if keyword.is_empty() {
        "card".to_string()
    } else {
        SlugService::slugify(keyword)
    };
    (topic_slug, keyword_slug)
}

/// Stateless slug generation utilities.
pub struct SlugService;

impl SlugService {
    /// Convert arbitrary text to a URL-friendly slug.
    pub fn slugify(text: &str) -> String {
        // NFKD normalize, strip combining marks, replace ß
        let normalized: String = text
            .nfkd()
            .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
            .collect();
        let lower = normalized.replace('ß', "s").to_lowercase();

        // Replace separators with hyphens, remove non-[a-z0-9-]
        let mut result = String::with_capacity(lower.len());
        for ch in lower.chars() {
            if ch.is_ascii_alphanumeric() {
                result.push(ch);
            } else if matches!(ch, ' ' | '_' | '.' | '/' | '\\' | '-') {
                result.push('-');
            }
        }

        // Collapse multiple hyphens
        let collapsed = collapse_hyphens(&result);

        // Strip leading/trailing hyphens
        let trimmed = collapsed.trim_matches('-');

        // Truncate to MAX_COMPONENT_LENGTH at word boundary
        if trimmed.len() <= MAX_COMPONENT_LENGTH {
            return trimmed.to_string();
        }

        // Find last hyphen before MAX_COMPONENT_LENGTH
        let truncated = &trimmed[..MAX_COMPONENT_LENGTH];
        if let Some(pos) = truncated.rfind('-') {
            truncated[..pos].to_string()
        } else {
            truncated.to_string()
        }
    }

    /// SHA-256[:length] of content. length must be in 1..=64.
    pub fn compute_hash(content: &str, length: usize) -> Result<String, SlugError> {
        if length == 0 || length > 64 {
            return Err(SlugError::InvalidHashLength(length));
        }
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        let hash = format!("{:x}", hasher.finalize());
        Ok(hash[..length].to_string())
    }

    /// Generate slug: "{topic}-{keyword}-{index}-{lang}".
    pub fn generate_slug(
        topic: &str,
        keyword: &str,
        index: u32,
        lang: &str,
    ) -> Result<String, SlugError> {
        let lang_lower = validate_lang(lang)?;
        let (topic_slug, keyword_slug) = slugify_components(topic, keyword);

        let suffix = format!("-{index}-{lang_lower}");
        let available = MAX_SLUG_LENGTH.saturating_sub(suffix.len());

        let full_base = format!("{topic_slug}-{keyword_slug}");
        let base = if full_base.len() > available {
            let half = available / 2;
            let t = Self::truncate_at_boundary(&topic_slug, half);
            let k = Self::truncate_at_boundary(&keyword_slug, half);
            format!("{t}-{k}")
        } else {
            full_base
        };

        Ok(format!("{base}{suffix}"))
    }

    /// Generate slug base without language suffix: "{topic}-{keyword}-{index}".
    pub fn generate_slug_base(topic: &str, keyword: &str, index: u32) -> Result<String, SlugError> {
        let (topic_slug, keyword_slug) = slugify_components(topic, keyword);
        Ok(format!("{topic_slug}-{keyword_slug}-{index}"))
    }

    /// Deterministic GUID: SHA-256[:32] of "{slug}:{source_path}".
    pub fn generate_deterministic_guid(slug: &str, source_path: &str) -> Result<String, SlugError> {
        if slug.is_empty() {
            return Err(SlugError::EmptyInput { field: "slug" });
        }
        if source_path.is_empty() {
            return Err(SlugError::EmptyInput {
                field: "source_path",
            });
        }
        let content = format!("{slug}:{source_path}");
        Self::compute_hash(&content, 32)
    }

    /// Extract components from a slug -> { topic, keyword, index, lang }.
    pub fn extract_components(slug: &str) -> SlugComponents {
        if slug.is_empty() {
            return SlugComponents::default();
        }

        let parts: Vec<&str> = slug.split('-').collect();
        if parts.len() < 3 {
            return SlugComponents {
                topic: slug.to_string(),
                ..Default::default()
            };
        }

        // Parse from the right: last = lang, second-to-last = index
        let lang_candidate = parts[parts.len() - 1];
        let index_candidate = parts[parts.len() - 2];

        let lang = if lang_candidate.len() == 2
            && lang_candidate.chars().all(|c| c.is_ascii_alphabetic())
        {
            lang_candidate.to_string()
        } else {
            return SlugComponents {
                topic: slug.to_string(),
                ..Default::default()
            };
        };

        let index = if let Ok(idx) = index_candidate.parse::<u32>() {
            idx
        } else {
            return SlugComponents {
                topic: slug.to_string(),
                ..Default::default()
            };
        };

        // Remaining parts are topic + keyword
        let remaining = &parts[..parts.len() - 2];
        let (topic, keyword) = if remaining.is_empty() {
            (String::new(), String::new())
        } else if remaining.len() == 1 {
            (remaining[0].to_string(), String::new())
        } else {
            let mid = remaining.len() / 2;
            let topic = remaining[..mid].join("-");
            let keyword = remaining[mid..].join("-");
            (topic, keyword)
        };

        SlugComponents {
            topic,
            keyword,
            index,
            lang,
        }
    }

    /// Validate: >= 3 chars, matches [a-z0-9][a-z0-9-]*[a-z0-9], no "--",
    /// ends with "-{2-letter-lang}".
    pub fn is_valid_slug(slug: &str) -> bool {
        if slug.len() < 3 {
            return false;
        }
        // Only lowercase alphanumeric and hyphens
        if !slug
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        {
            return false;
        }
        // No leading/trailing hyphens
        if slug.starts_with('-') || slug.ends_with('-') {
            return false;
        }
        // No double hyphens
        if slug.contains("--") {
            return false;
        }
        // Must end with "-{2-letter-lang}"
        if let Some(pos) = slug.rfind('-') {
            let lang = &slug[pos + 1..];
            lang.len() == 2 && lang.chars().all(|c| c.is_ascii_alphabetic())
        } else {
            false
        }
    }

    /// SHA-256[:12] of "{front.trim()}|{back.trim()}".
    pub fn compute_content_hash(front: &str, back: &str) -> String {
        let content = format!("{}|{}", front.trim(), back.trim());
        // unwrap safe: length 12 is valid
        Self::compute_hash(&content, 12).expect("length 12 is valid")
    }

    /// SHA-256[:6] of "{note_type}|{sorted,tags}".
    pub fn compute_metadata_hash(note_type: &str, tags: &[String]) -> String {
        let mut sorted_tags: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
        sorted_tags.sort();
        let content = format!("{}|{}", note_type, sorted_tags.join(","));
        // unwrap safe: length 6 is valid
        Self::compute_hash(&content, 6).expect("length 6 is valid")
    }

    /// Truncate a string at a word boundary (hyphen), stripping trailing hyphens.
    fn truncate_at_boundary(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            return s.to_string();
        }
        let truncated = &s[..max_len];
        if let Some(pos) = truncated.rfind('-') {
            truncated[..pos].trim_end_matches('-').to_string()
        } else {
            truncated.trim_end_matches('-').to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Constants ==========

    #[test]
    fn max_component_length_is_50() {
        assert_eq!(MAX_COMPONENT_LENGTH, 50);
    }

    #[test]
    fn max_slug_length_is_100() {
        assert_eq!(MAX_SLUG_LENGTH, 100);
    }

    // ========== slugify ==========

    #[test]
    fn slugify_simple_text() {
        assert_eq!(SlugService::slugify("Hello World"), "hello-world");
    }

    #[test]
    fn slugify_empty_string() {
        assert_eq!(SlugService::slugify(""), "");
    }

    #[test]
    fn slugify_unicode_nfkd_normalization() {
        // "café" -> "cafe" (strip combining acute accent)
        assert_eq!(SlugService::slugify("café"), "cafe");
    }

    #[test]
    fn slugify_accented_characters() {
        assert_eq!(SlugService::slugify("über straße"), "uber-strase");
    }

    #[test]
    fn slugify_replaces_separators_with_hyphens() {
        // spaces, underscores, dots, slashes -> hyphens
        assert_eq!(
            SlugService::slugify("hello_world.test/path\\here"),
            "hello-world-test-path-here"
        );
    }

    #[test]
    fn slugify_removes_non_alphanumeric() {
        assert_eq!(SlugService::slugify("hello!@#$%world"), "helloworld");
    }

    #[test]
    fn slugify_collapses_multiple_hyphens() {
        assert_eq!(SlugService::slugify("hello---world"), "hello-world");
    }

    #[test]
    fn slugify_strips_leading_trailing_hyphens() {
        assert_eq!(SlugService::slugify("-hello-world-"), "hello-world");
    }

    #[test]
    fn slugify_lowercase() {
        assert_eq!(SlugService::slugify("HELLO WORLD"), "hello-world");
    }

    #[test]
    fn slugify_numbers_preserved() {
        assert_eq!(SlugService::slugify("chapter 42"), "chapter-42");
    }

    #[test]
    fn slugify_truncates_to_max_component_length() {
        let long_text = "a".repeat(60);
        let result = SlugService::slugify(&long_text);
        assert!(result.len() <= MAX_COMPONENT_LENGTH);
    }

    #[test]
    fn slugify_truncates_at_word_boundary() {
        // Create text that exceeds 50 chars with hyphens for word boundaries
        let text = "alpha-bravo-charlie-delta-echo-foxtrot-golf-hotel-india-juliet";
        let result = SlugService::slugify(text);
        assert!(result.len() <= MAX_COMPONENT_LENGTH);
        // Should not end with a hyphen
        assert!(!result.ends_with('-'));
    }

    #[test]
    fn slugify_japanese_characters_stripped() {
        // Non-latin characters that don't decompose to ASCII are removed
        let result = SlugService::slugify("日本語テスト");
        assert!(
            result.is_empty()
                || result
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-')
        );
    }

    #[test]
    fn slugify_mixed_unicode_and_ascii() {
        let result = SlugService::slugify("Python décorateurs");
        assert_eq!(result, "python-decorateurs");
    }

    // ========== compute_hash ==========

    #[test]
    fn compute_hash_default_length_6() {
        let result = SlugService::compute_hash("hello", 6).unwrap();
        assert_eq!(result.len(), 6);
        assert!(result.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn compute_hash_length_1() {
        let result = SlugService::compute_hash("hello", 1).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn compute_hash_length_64() {
        let result = SlugService::compute_hash("hello", 64).unwrap();
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn compute_hash_length_0_errors() {
        let result = SlugService::compute_hash("hello", 0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SlugError::InvalidHashLength(0)
        ));
    }

    #[test]
    fn compute_hash_length_65_errors() {
        let result = SlugService::compute_hash("hello", 65);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SlugError::InvalidHashLength(65)
        ));
    }

    #[test]
    fn compute_hash_deterministic() {
        let h1 = SlugService::compute_hash("test content", 6).unwrap();
        let h2 = SlugService::compute_hash("test content", 6).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn compute_hash_different_content_different_hash() {
        let h1 = SlugService::compute_hash("hello", 6).unwrap();
        let h2 = SlugService::compute_hash("world", 6).unwrap();
        assert_ne!(h1, h2);
    }

    // ========== generate_slug ==========

    #[test]
    fn generate_slug_basic_format() {
        let slug = SlugService::generate_slug("python", "decorators", 1, "en").unwrap();
        assert_eq!(slug, "python-decorators-1-en");
    }

    #[test]
    fn generate_slug_slugifies_components() {
        let slug = SlugService::generate_slug("Hello World", "My Topic", 0, "en").unwrap();
        assert_eq!(slug, "hello-world-my-topic-0-en");
    }

    #[test]
    fn generate_slug_empty_topic_becomes_untitled() {
        let slug = SlugService::generate_slug("", "keyword", 0, "en").unwrap();
        assert!(slug.starts_with("untitled-"));
    }

    #[test]
    fn generate_slug_empty_keyword_becomes_card() {
        let slug = SlugService::generate_slug("topic", "", 0, "en").unwrap();
        assert!(slug.contains("-card-"));
    }

    #[test]
    fn generate_slug_invalid_lang_too_short() {
        let result = SlugService::generate_slug("topic", "keyword", 0, "e");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SlugError::InvalidLang(_)));
    }

    #[test]
    fn generate_slug_invalid_lang_too_long() {
        let result = SlugService::generate_slug("topic", "keyword", 0, "eng");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SlugError::InvalidLang(_)));
    }

    #[test]
    fn generate_slug_invalid_lang_empty() {
        let result = SlugService::generate_slug("topic", "keyword", 0, "");
        assert!(result.is_err());
    }

    #[test]
    fn generate_slug_stays_within_max_length() {
        let long_topic = "a".repeat(80);
        let long_keyword = "b".repeat(80);
        let slug = SlugService::generate_slug(&long_topic, &long_keyword, 0, "en").unwrap();
        assert!(
            slug.len() <= MAX_SLUG_LENGTH,
            "slug too long: {} chars",
            slug.len()
        );
    }

    #[test]
    fn generate_slug_truncation_strips_trailing_hyphens() {
        let long_topic = "alpha-bravo-charlie-delta-echo-foxtrot-golf-hotel";
        let long_keyword = "india-juliet-kilo-lima-mike-november-oscar-papa";
        let slug = SlugService::generate_slug(long_topic, long_keyword, 999, "en").unwrap();
        assert!(slug.len() <= MAX_SLUG_LENGTH);
        // No double hyphens
        assert!(!slug.contains("--"), "slug contains double hyphens: {slug}");
    }

    #[test]
    fn generate_slug_lang_lowercased() {
        let slug = SlugService::generate_slug("topic", "keyword", 0, "EN").unwrap();
        assert!(slug.ends_with("-en"));
    }

    // ========== generate_slug_base ==========

    #[test]
    fn generate_slug_base_format() {
        let base = SlugService::generate_slug_base("python", "decorators", 1).unwrap();
        assert_eq!(base, "python-decorators-1");
    }

    #[test]
    fn generate_slug_base_empty_topic_becomes_untitled() {
        let base = SlugService::generate_slug_base("", "keyword", 0).unwrap();
        assert!(base.starts_with("untitled-"));
    }

    #[test]
    fn generate_slug_base_empty_keyword_becomes_card() {
        let base = SlugService::generate_slug_base("topic", "", 0).unwrap();
        assert!(base.contains("-card-"));
    }

    #[test]
    fn generate_slug_base_no_lang_suffix() {
        let base = SlugService::generate_slug_base("topic", "keyword", 0).unwrap();
        // Should not end with "-xx" where xx is a 2-letter lang
        assert_eq!(base, "topic-keyword-0");
    }

    // ========== generate_deterministic_guid ==========

    #[test]
    fn deterministic_guid_length_32() {
        let guid =
            SlugService::generate_deterministic_guid("my-slug-0-en", "notes/test.md").unwrap();
        assert_eq!(guid.len(), 32);
        assert!(guid.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn deterministic_guid_is_deterministic() {
        let g1 = SlugService::generate_deterministic_guid("slug-0-en", "path.md").unwrap();
        let g2 = SlugService::generate_deterministic_guid("slug-0-en", "path.md").unwrap();
        assert_eq!(g1, g2);
    }

    #[test]
    fn deterministic_guid_different_inputs_different_guids() {
        let g1 = SlugService::generate_deterministic_guid("slug-a-0-en", "path.md").unwrap();
        let g2 = SlugService::generate_deterministic_guid("slug-b-0-en", "path.md").unwrap();
        assert_ne!(g1, g2);
    }

    #[test]
    fn deterministic_guid_empty_slug_errors() {
        let result = SlugService::generate_deterministic_guid("", "path.md");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SlugError::EmptyInput { field: "slug" }
        ));
    }

    #[test]
    fn deterministic_guid_empty_source_path_errors() {
        let result = SlugService::generate_deterministic_guid("my-slug-0-en", "");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SlugError::EmptyInput {
                field: "source_path"
            }
        ));
    }

    // ========== extract_components ==========

    #[test]
    fn extract_components_standard_slug() {
        let c = SlugService::extract_components("python-decorators-1-en");
        assert_eq!(c.lang, "en");
        assert_eq!(c.index, 1);
        assert_eq!(c.topic, "python");
        assert_eq!(c.keyword, "decorators");
    }

    #[test]
    fn extract_components_multi_word_topic() {
        let c = SlugService::extract_components("hello-world-my-topic-0-ru");
        assert_eq!(c.lang, "ru");
        assert_eq!(c.index, 0);
        // Remaining "hello-world-my-topic" split in half: 4 parts -> 2+2
        assert!(!c.topic.is_empty());
        assert!(!c.keyword.is_empty());
    }

    #[test]
    fn extract_components_empty_string() {
        let c = SlugService::extract_components("");
        assert_eq!(c.topic, "");
        assert_eq!(c.keyword, "");
        assert_eq!(c.index, 0);
        assert_eq!(c.lang, "");
    }

    #[test]
    fn extract_components_roundtrip() {
        let slug = SlugService::generate_slug("python", "decorators", 1, "en").unwrap();
        let c = SlugService::extract_components(&slug);
        assert_eq!(c.lang, "en");
        assert_eq!(c.index, 1);
        // Topic and keyword should reconstruct to original slugified forms
        assert_eq!(c.topic, "python");
        assert_eq!(c.keyword, "decorators");
    }

    #[test]
    fn extract_components_single_word_remaining() {
        // If only one word remains after lang+index, topic gets it, keyword empty
        let c = SlugService::extract_components("topic-0-en");
        assert_eq!(c.lang, "en");
        assert_eq!(c.index, 0);
        assert_eq!(c.topic, "topic");
        assert_eq!(c.keyword, "");
    }

    // ========== is_valid_slug ==========

    #[test]
    fn is_valid_slug_valid() {
        assert!(SlugService::is_valid_slug("python-decorators-1-en"));
    }

    #[test]
    fn is_valid_slug_rejects_empty() {
        assert!(!SlugService::is_valid_slug(""));
    }

    #[test]
    fn is_valid_slug_rejects_too_short() {
        assert!(!SlugService::is_valid_slug("ab"));
    }

    #[test]
    fn is_valid_slug_rejects_uppercase() {
        assert!(!SlugService::is_valid_slug("Hello-World-0-en"));
    }

    #[test]
    fn is_valid_slug_rejects_double_hyphens() {
        assert!(!SlugService::is_valid_slug("hello--world-0-en"));
    }

    #[test]
    fn is_valid_slug_rejects_leading_hyphen() {
        assert!(!SlugService::is_valid_slug("-hello-world-0-en"));
    }

    #[test]
    fn is_valid_slug_rejects_trailing_hyphen() {
        assert!(!SlugService::is_valid_slug("hello-world-0-en-"));
    }

    #[test]
    fn is_valid_slug_rejects_no_lang_suffix() {
        // Must end with "-{2-letter-lang}"
        assert!(!SlugService::is_valid_slug("hello-world-123"));
    }

    #[test]
    fn is_valid_slug_rejects_special_characters() {
        assert!(!SlugService::is_valid_slug("hello_world-0-en"));
    }

    #[test]
    fn is_valid_slug_minimal_valid() {
        // Minimum: 3 chars, starts/ends alphanumeric, has lang suffix
        assert!(SlugService::is_valid_slug("a-0-en"));
    }

    // ========== compute_content_hash ==========

    #[test]
    fn compute_content_hash_length_12() {
        let h = SlugService::compute_content_hash("front text", "back text");
        assert_eq!(h.len(), 12);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn compute_content_hash_trims_whitespace() {
        let h1 = SlugService::compute_content_hash("  front  ", "  back  ");
        let h2 = SlugService::compute_content_hash("front", "back");
        assert_eq!(h1, h2);
    }

    #[test]
    fn compute_content_hash_deterministic() {
        let h1 = SlugService::compute_content_hash("front", "back");
        let h2 = SlugService::compute_content_hash("front", "back");
        assert_eq!(h1, h2);
    }

    #[test]
    fn compute_content_hash_different_content_different_hash() {
        let h1 = SlugService::compute_content_hash("front1", "back1");
        let h2 = SlugService::compute_content_hash("front2", "back2");
        assert_ne!(h1, h2);
    }

    // ========== compute_metadata_hash ==========

    #[test]
    fn compute_metadata_hash_length_6() {
        let h = SlugService::compute_metadata_hash("Basic", &["tag1".into(), "tag2".into()]);
        assert_eq!(h.len(), 6);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn compute_metadata_hash_tag_order_independent() {
        let h1 = SlugService::compute_metadata_hash("Basic", &["beta".into(), "alpha".into()]);
        let h2 = SlugService::compute_metadata_hash("Basic", &["alpha".into(), "beta".into()]);
        assert_eq!(h1, h2);
    }

    #[test]
    fn compute_metadata_hash_deterministic() {
        let tags: Vec<String> = vec!["a".into(), "b".into()];
        let h1 = SlugService::compute_metadata_hash("Cloze", &tags);
        let h2 = SlugService::compute_metadata_hash("Cloze", &tags);
        assert_eq!(h1, h2);
    }

    #[test]
    fn compute_metadata_hash_different_note_type_different_hash() {
        let tags: Vec<String> = vec!["tag".into()];
        let h1 = SlugService::compute_metadata_hash("Basic", &tags);
        let h2 = SlugService::compute_metadata_hash("Cloze", &tags);
        assert_ne!(h1, h2);
    }

    #[test]
    fn compute_metadata_hash_empty_tags() {
        let h = SlugService::compute_metadata_hash("Basic", &[]);
        assert_eq!(h.len(), 6);
    }

    // ========== Send + Sync ==========

    #[test]
    fn slug_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SlugService>();
        assert_send_sync::<SlugComponents>();
        assert_send_sync::<SlugError>();
    }
}

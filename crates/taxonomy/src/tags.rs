/// Standard double-colon tag prefixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TagPrefix {
    Android,
    Kotlin,
    Cs,
    Topic,
    Difficulty,
    Lang,
    Source,
    Context,
    Bias,
    Testing,
    Architecture,
    Performance,
    Platform,
    Security,
    Networking,
}

impl TagPrefix {
    /// Return the lowercase string form (e.g. `"android"`, `"cs"`).
    pub fn as_str(&self) -> &'static str {
        todo!()
    }
}

/// All valid prefix strings as a static slice.
pub static VALID_PREFIXES: &[&str] = &[];

/// Prefixes that denote meta tags.
pub static META_TAG_PREFIXES: &[&str] = &[];

/// Standalone meta tag values.
pub static META_TAGS: &[&str] = &[];

/// Number of entries in the tag mapping.
pub const TAG_MAPPING_LEN: usize = 0;

/// Variant -> normalized canonical tag lookup.
pub fn lookup_tag(_variant: &str) -> Option<&'static str> {
    None
}

/// All canonical Kotlin topic tags.
pub static KOTLIN_TOPIC_TAGS: &[&str] = &[];

/// All canonical Android topic tags.
pub static ANDROID_TOPIC_TAGS: &[&str] = &[];

/// All canonical CompSci topic tags.
pub static COMPSCI_TOPIC_TAGS: &[&str] = &[];

/// All canonical Cognitive Bias topic tags.
pub static COGNITIVE_BIAS_TOPIC_TAGS: &[&str] = &[];

/// Check if a tag is a known topic tag (O(1) lookup).
pub fn is_known_topic_tag(_tag: &str) -> bool {
    false
}

/// Valid difficulty meta tag values.
pub static VALID_DIFFICULTIES: &[&str] = &[];

/// Valid language meta tag values.
pub static VALID_LANGS: &[&str] = &[];

/// Topic prefixes in underscore format used internally.
pub(crate) static TOPIC_PREFIXES: &[&str] = &[];

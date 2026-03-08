pub mod normalize;
pub mod tags;

// Re-export key functions at crate root.
pub use normalize::{
    is_meta_tag, is_topic_tag, normalize_tag, normalize_tags, suggest_tag, validate_tag,
};
pub use tags::{TagPrefix, VALID_PREFIXES, is_known_topic_tag, lookup_tag};

#[cfg(test)]
mod tests_tags;

#[cfg(test)]
mod tests_normalize;

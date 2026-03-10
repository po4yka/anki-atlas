#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use card::slug::{MAX_COMPONENT_LENGTH, MAX_SLUG_LENGTH, SlugService};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct SlugSeed<'a> {
    payload: &'a [u8],
}

#[derive(Debug)]
struct SlugInput {
    lang: String,
    index: u32,
    hash_length: usize,
    topic: String,
    keyword: String,
    source_path: String,
    note_type: String,
    tags: Vec<String>,
    content: String,
}

impl SlugInput {
    fn from_bytes(bytes: &[u8]) -> Self {
        let raw = String::from_utf8_lossy(bytes).into_owned();
        let mut lines = raw.lines();
        let header = lines.next().unwrap_or_default();
        let topic = lines.next().unwrap_or_default().to_string();
        let keyword = lines.next().unwrap_or_default().to_string();
        let source_path = lines.next().unwrap_or_default().to_string();
        let note_type = lines.next().unwrap_or("Basic").to_string();
        let tags_line = lines.next().unwrap_or_default();
        let content = lines.collect::<Vec<_>>().join("\n");

        let lang = header
            .split_whitespace()
            .find_map(|part| part.strip_prefix("lang="))
            .unwrap_or_default()
            .to_string();
        let index = header
            .split_whitespace()
            .find_map(|part| part.strip_prefix("index="))
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or_default();
        let hash_length = header
            .split_whitespace()
            .find_map(|part| part.strip_prefix("hash_len="))
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(12);
        let tags = tags_line
            .split(',')
            .map(str::trim)
            .filter(|tag| !tag.is_empty())
            .map(ToString::to_string)
            .collect();

        Self {
            lang,
            index,
            hash_length,
            topic,
            keyword,
            source_path,
            note_type,
            tags,
            content: if content.is_empty() { raw } else { content },
        }
    }
}

fn is_slug_component(text: &str) -> bool {
    text.chars()
        .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-')
}

fuzz_target!(|data: &[u8]| {
    let Ok(seed) = SlugSeed::arbitrary_take_rest(Unstructured::new(data)) else {
        return;
    };
    let input = SlugInput::from_bytes(seed.payload);

    let slugified = SlugService::slugify(&input.content);
    assert_eq!(slugified, SlugService::slugify(&input.content));
    assert!(slugified.len() <= MAX_COMPONENT_LENGTH);
    assert!(is_slug_component(&slugified));
    assert!(!slugified.starts_with('-'));
    assert!(!slugified.ends_with('-'));
    assert!(!slugified.contains("--"));

    let valid_hash_len = input.hash_length.clamp(1, 64);
    let hash = SlugService::compute_hash(&input.content, valid_hash_len).unwrap();
    assert_eq!(hash.len(), valid_hash_len);
    assert!(hash.chars().all(|ch| ch.is_ascii_hexdigit()));

    let content_hash = SlugService::compute_content_hash(&input.topic, &input.keyword);
    assert_eq!(content_hash.len(), 12);
    assert!(content_hash.chars().all(|ch| ch.is_ascii_hexdigit()));

    let metadata_hash = SlugService::compute_metadata_hash(&input.note_type, &input.tags);
    assert_eq!(metadata_hash.len(), 6);
    assert!(metadata_hash.chars().all(|ch| ch.is_ascii_hexdigit()));

    if input.lang.len() == 2 && input.lang.chars().all(|ch| ch.is_ascii_alphabetic()) {
        let slug =
            SlugService::generate_slug(&input.topic, &input.keyword, input.index, &input.lang)
                .unwrap();
        assert!(slug.len() <= MAX_SLUG_LENGTH);
        assert!(SlugService::is_valid_slug(&slug));

        let slug_base =
            SlugService::generate_slug_base(&input.topic, &input.keyword, input.index).unwrap();
        let components = SlugService::extract_components(&slug);
        assert_eq!(components.index, input.index);
        assert_eq!(components.lang, input.lang.to_lowercase());
        assert!(!slug_base.is_empty());

        let source_path = if input.source_path.is_empty() {
            "notes/default.md"
        } else {
            input.source_path.as_str()
        };
        let guid = SlugService::generate_deterministic_guid(&slug, source_path).unwrap();
        assert_eq!(guid.len(), 32);
        assert!(guid.chars().all(|ch| ch.is_ascii_hexdigit()));
    }
});

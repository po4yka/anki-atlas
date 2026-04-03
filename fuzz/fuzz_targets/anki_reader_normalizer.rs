#![no_main]

use anki_reader::normalizer::{
    CodeHandling, FieldRole, classify_field, normalize_whitespace, strip_html,
};
use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct NormalizerSeed<'a> {
    payload: &'a [u8],
}

#[derive(Debug)]
struct NormalizerInput {
    preserve_code: CodeHandling,
    field_name: String,
    text: String,
}

impl NormalizerInput {
    fn from_bytes(bytes: &[u8]) -> Self {
        let raw = String::from_utf8_lossy(bytes).into_owned();
        let mut lines = raw.lines();
        let header = lines.next().unwrap_or_default();
        let text = lines.collect::<Vec<_>>().join("\n");

        let preserve_code = if header.contains("pc=1") {
            CodeHandling::Preserve
        } else {
            CodeHandling::Strip
        };
        let field_name = header
            .split_whitespace()
            .find_map(|part| part.strip_prefix("field="))
            .unwrap_or(header)
            .to_string();
        let text = if text.is_empty() { raw } else { text };

        Self {
            preserve_code,
            field_name,
            text,
        }
    }
}

fn contains_html_like_tag(text: &str) -> bool {
    let bytes = text.as_bytes();
    for start in 0..bytes.len() {
        if bytes[start] != b'<' {
            continue;
        }

        let Some(end_offset) = bytes[start + 1..].iter().position(|b| *b == b'>') else {
            continue;
        };
        let end = start + 1 + end_offset;
        if end == start + 1 || end - start > 32 {
            continue;
        }

        let inner = &text[start + 1..end];
        if inner
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '!' | '-' | '_' | ' '))
        {
            return true;
        }
    }

    false
}

fn is_simple_tag_only_input(text: &str) -> bool {
    let trimmed = text.trim();
    !trimmed.is_empty()
        && trimmed.starts_with('<')
        && trimmed.ends_with('>')
        && !trimmed.contains('&')
}

fuzz_target!(|data: &[u8]| {
    let Ok(seed) = NormalizerSeed::arbitrary_take_rest(Unstructured::new(data)) else {
        return;
    };
    let input = NormalizerInput::from_bytes(seed.payload);

    let stripped = strip_html(&input.text, input.preserve_code);
    let stripped_again = strip_html(&input.text, input.preserve_code);
    assert_eq!(stripped, stripped_again);

    let normalized = normalize_whitespace(&stripped);
    assert_eq!(normalized, normalize_whitespace(&normalized));

    let class = classify_field(&input.field_name);
    assert!(matches!(class, FieldRole::Front | FieldRole::Back | FieldRole::Extra | FieldRole::Other));

    if input.preserve_code == CodeHandling::Strip && is_simple_tag_only_input(&input.text) {
        assert!(!contains_html_like_tag(&normalized));
    }
});

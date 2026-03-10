#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use generator::apf::linter::validate_apf;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct ApfSeed<'a> {
    payload: &'a [u8],
}

#[derive(Debug)]
struct ApfInput {
    expected_slug: Option<String>,
    html: String,
}

impl ApfInput {
    fn from_bytes(bytes: &[u8]) -> Self {
        let raw = String::from_utf8_lossy(bytes).into_owned();
        let mut lines = raw.lines();
        let header = lines.next().unwrap_or_default();
        let html = lines.collect::<Vec<_>>().join("\n");

        let expected_slug = header
            .split_whitespace()
            .find_map(|part| part.strip_prefix("slug="))
            .filter(|slug| !slug.is_empty())
            .map(ToString::to_string);
        let html = if html.is_empty() { raw } else { html };

        Self {
            expected_slug,
            html,
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let Ok(seed) = ApfSeed::arbitrary_take_rest(Unstructured::new(data)) else {
        return;
    };
    let input = ApfInput::from_bytes(seed.payload);

    let result = validate_apf(&input.html, input.expected_slug.as_deref());
    let result_again = validate_apf(&input.html, input.expected_slug.as_deref());

    assert_eq!(result.errors, result_again.errors);
    assert_eq!(result.warnings, result_again.warnings);
    assert_eq!(result.is_valid(), result.errors.is_empty());
    assert!(result
        .errors
        .iter()
        .chain(result.warnings.iter())
        .all(|message| !message.trim().is_empty()));

    if input.html.is_empty() {
        assert!(!result.is_valid());
    }
});

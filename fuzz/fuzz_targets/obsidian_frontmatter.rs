#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use obsidian::{fuzzing::split_frontmatter, parse_frontmatter, write_frontmatter};

#[derive(Arbitrary, Debug)]
struct FrontmatterSeed<'a> {
    payload: &'a [u8],
}

#[derive(Debug)]
struct FrontmatterInput {
    content: String,
    body: String,
}

impl FrontmatterInput {
    fn from_bytes(bytes: &[u8]) -> Self {
        let raw = String::from_utf8_lossy(bytes).into_owned();
        let (content, body) = raw
            .split_once("\n--BODY--\n")
            .map(|(content, body)| (content.to_string(), body.to_string()))
            .unwrap_or_else(|| (raw.clone(), "Generated fuzz body".to_string()));

        Self { content, body }
    }
}

fuzz_target!(|data: &[u8]| {
    let Ok(seed) = FrontmatterSeed::arbitrary_take_rest(Unstructured::new(data)) else {
        return;
    };
    let input = FrontmatterInput::from_bytes(seed.payload);

    let _ = split_frontmatter(&input.content);

    if let Ok(parsed) = parse_frontmatter(&input.content) {
        if let Ok(rewritten) = write_frontmatter(&parsed, &input.body) {
            let reparsed = parse_frontmatter(&rewritten).unwrap();
            assert_eq!(parsed, reparsed);

            let (yaml, _) = split_frontmatter(&rewritten);
            assert!(yaml.is_some());
        }
    }
});

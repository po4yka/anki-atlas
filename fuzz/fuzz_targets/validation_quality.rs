#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use validation::assess_quality;

#[derive(Arbitrary, Debug)]
struct QualitySeed<'a> {
    payload: &'a [u8],
}

#[derive(Debug)]
struct QualityInput {
    front: String,
    back: String,
}

impl QualityInput {
    fn from_bytes(bytes: &[u8]) -> Self {
        let raw = String::from_utf8_lossy(bytes).into_owned();
        let (front, back) = raw
            .split_once("\n--BACK--\n")
            .map(|(front, back)| (front.to_string(), back.to_string()))
            .unwrap_or_else(|| (raw.clone(), raw));

        Self { front, back }
    }
}

fn in_unit_interval(value: f64) -> bool {
    (0.0..=1.0).contains(&value)
}

fuzz_target!(|data: &[u8]| {
    let Ok(seed) = QualitySeed::arbitrary_take_rest(Unstructured::new(data)) else {
        return;
    };
    let input = QualityInput::from_bytes(seed.payload);

    let score = assess_quality(&input.front, &input.back);
    let score_again = assess_quality(&input.front, &input.back);
    let trimmed_score = assess_quality(input.front.trim(), input.back.trim());

    assert_eq!(score, score_again);
    assert_eq!(score, trimmed_score);
    assert!(in_unit_interval(score.clarity));
    assert!(in_unit_interval(score.atomicity));
    assert!(in_unit_interval(score.testability));
    assert!(in_unit_interval(score.memorability));
    assert!(in_unit_interval(score.accuracy));
    assert!(in_unit_interval(score.relevance));
    assert!(in_unit_interval(score.overall()));

    let expected_overall = (score.clarity
        + score.atomicity
        + score.testability
        + score.memorability
        + score.accuracy
        + 1.5 * score.relevance)
        / 6.5;
    assert!((score.overall() - expected_overall).abs() <= f64::EPSILON);
});

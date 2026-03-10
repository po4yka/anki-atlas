#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use indexer::qdrant::QdrantRepository;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct SparseVectorSeed<'a> {
    payload: &'a [u8],
}

fn has_tokenizable_content(text: &str) -> bool {
    text.split_whitespace()
        .map(|token| token.chars().filter(|ch| ch.is_alphanumeric()).count())
        .any(|len| len > 0)
}

fuzz_target!(|data: &[u8]| {
    let Ok(seed) = SparseVectorSeed::arbitrary_take_rest(Unstructured::new(data)) else {
        return;
    };
    let input = String::from_utf8_lossy(seed.payload).into_owned();

    let vector = QdrantRepository::text_to_sparse_vector(&input);
    let vector_again = QdrantRepository::text_to_sparse_vector(&input);

    assert_eq!(vector, vector_again);
    assert_eq!(vector.indices.len(), vector.values.len());
    assert!(vector
        .values
        .iter()
        .all(|value| value.is_finite() && *value > 0.0));
    assert!(vector.indices.windows(2).all(|pair| pair[0] < pair[1]));

    if vector.values.is_empty() {
        assert!(!has_tokenizable_content(&input));
    } else {
        let norm = vector
            .values
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() <= 1e-4);
    }
});

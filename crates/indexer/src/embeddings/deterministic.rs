use async_trait::async_trait;
use sha2::{Digest, Sha256};

use super::{EmbeddingError, EmbeddingInput, EmbeddingProvider};

/// Mock embedding provider for tests. Returns deterministic vectors from MD5 hash.
#[derive(Debug, Clone)]
pub struct DeterministicEmbeddingProvider {
    dimension: usize,
}

impl DeterministicEmbeddingProvider {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl EmbeddingProvider for DeterministicEmbeddingProvider {
    fn model_name(&self) -> &str {
        "mock/test"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed_inputs(
        &self,
        inputs: &[EmbeddingInput],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            let mut hasher = Sha256::new();
            hasher.update(format!("{:?}", input.task).as_bytes());
            hasher.update(input.title.as_deref().unwrap_or_default().as_bytes());
            hasher.update(
                input
                    .output_dimensionality
                    .map(|dim| dim.to_string())
                    .unwrap_or_default()
                    .as_bytes(),
            );
            for part in &input.parts {
                hasher.update([0xff]);
                part.append_hash_bytes(&mut hasher);
            }
            let hash_bytes = hasher.finalize();
            let mut vec = Vec::with_capacity(self.dimension);
            for i in 0..self.dimension {
                let byte = hash_bytes[i % 32];
                let val = (f32::from(byte) / 127.5) - 1.0;
                vec.push(val);
            }
            results.push(vec);
        }
        Ok(results)
    }
}

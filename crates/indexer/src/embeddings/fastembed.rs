use std::sync::Mutex;

use async_trait::async_trait;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use tracing::{info, instrument};

use super::{EmbeddingError, EmbeddingInput, EmbeddingProvider};

/// Local embedding provider using fastembed (ONNX Runtime).
///
/// Downloads and caches models locally on first use. No API key required.
/// Runs inference on CPU -- suitable for moderate batch sizes.
pub struct FastEmbedProvider {
    model: Mutex<TextEmbedding>,
    model_name: String,
    dimension: usize,
}

impl FastEmbedProvider {
    /// Create a new provider with the specified model.
    ///
    /// Supported model strings (mapped to fastembed enum variants):
    /// - "all-MiniLM-L6-v2" (384d)
    /// - "all-MiniLM-L12-v2" (384d)
    /// - "bge-small-en-v1.5" (384d)
    /// - "bge-base-en-v1.5" (768d)
    /// - "bge-large-en-v1.5" (1024d)
    /// - "multilingual-e5-small" (384d)
    /// - "nomic-embed-text-v1.5" (768d)
    pub fn new(model_name: String, dimension: usize) -> Result<Self, EmbeddingError> {
        let model_enum = parse_model_name(&model_name)?;

        info!(model = %model_name, dim = dimension, "Initializing fastembed provider");

        let options = InitOptions::new(model_enum).with_show_download_progress(true);

        let text_embedding = TextEmbedding::try_new(options).map_err(|e| {
            EmbeddingError::NotConfigured(format!("fastembed init failed for {model_name}: {e}"))
        })?;

        Ok(Self {
            model: Mutex::new(text_embedding),
            model_name,
            dimension,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for FastEmbedProvider {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    #[instrument(skip(self, inputs), fields(count = inputs.len()))]
    async fn embed_inputs(
        &self,
        inputs: &[EmbeddingInput],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let texts: Vec<String> = inputs
            .iter()
            .filter_map(|input| {
                input.parts.iter().find_map(|part| match part {
                    super::EmbeddingPart::Text { text } => Some(text.clone()),
                    _ => None,
                })
            })
            .collect();

        if texts.is_empty() {
            return Err(EmbeddingError::UnsupportedInput {
                message: "fastembed only supports text inputs".into(),
            });
        }

        // Run sync fastembed on a blocking thread to avoid blocking the async runtime
        let model = self.model.lock().map_err(|e| EmbeddingError::BatchFailed {
            source: format!("mutex poisoned: {e}").into(),
        })?;

        // Clone the texts for the blocking closure
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let embeddings =
            // fastembed embed is CPU-bound; use block_in_place to run in current thread
            // without yielding the tokio runtime
            tokio::task::block_in_place(|| model.embed(text_refs, None)).map_err(|e| {
                EmbeddingError::BatchFailed {
                    source: format!("fastembed embed failed: {e}").into(),
                }
            })?;

        Ok(embeddings)
    }
}

/// Parse a human-readable model name string to a fastembed `EmbeddingModel` enum.
fn parse_model_name(name: &str) -> Result<EmbeddingModel, EmbeddingError> {
    match name.to_lowercase().as_str() {
        "all-minilm-l6-v2" => Ok(EmbeddingModel::AllMiniLML6V2),
        "all-minilm-l12-v2" => Ok(EmbeddingModel::AllMiniLML12V2),
        "bge-small-en-v1.5" | "bge-small-en" => Ok(EmbeddingModel::BGESmallENV15),
        "bge-base-en-v1.5" | "bge-base-en" => Ok(EmbeddingModel::BGEBaseENV15),
        "bge-large-en-v1.5" | "bge-large-en" => Ok(EmbeddingModel::BGELargeENV15),
        "multilingual-e5-small" => Ok(EmbeddingModel::MultilingualE5Small),
        "multilingual-e5-large" => Ok(EmbeddingModel::MultilingualE5Large),
        "nomic-embed-text-v1.5" => Ok(EmbeddingModel::NomicEmbedTextV15),
        _ => Err(EmbeddingError::NotConfigured(format!(
            "unknown fastembed model: {name}. Supported: all-MiniLM-L6-v2, bge-small-en-v1.5, multilingual-e5-small, etc."
        ))),
    }
}

impl std::fmt::Debug for FastEmbedProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedProvider")
            .field("model_name", &self.model_name)
            .field("dimension", &self.dimension)
            .finish()
    }
}

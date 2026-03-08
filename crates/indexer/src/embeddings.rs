use async_trait::async_trait;
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Errors from embedding operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("provider not configured: {0}")]
    NotConfigured(String),
    #[error("batch embedding failed: {source}")]
    BatchFailed {
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("rate limited, retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },
}

/// Trait for embedding providers. All impls must be Send + Sync.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Model identifier for version tracking.
    fn model_name(&self) -> &str;

    /// Dimensionality of output vectors.
    fn dimension(&self) -> usize;

    /// Embed a batch of texts. Returns one vector per input text.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
}

/// SHA-256[:16] hash of "{model_name}:{text}" for change detection.
pub fn content_hash(model_name: &str, text: &str) -> String {
    let input = format!("{model_name}:{text}");
    let hash = Sha256::digest(input.as_bytes());
    hex::encode(&hash[..8]) // 16 hex chars = 8 bytes
}

/// Mock embedding provider for tests. Returns deterministic vectors from MD5 hash.
#[derive(Debug, Clone)]
pub struct MockEmbeddingProvider {
    dim: usize,
}

impl MockEmbeddingProvider {
    pub fn new(dimension: usize) -> Self {
        Self { dim: dimension }
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    fn model_name(&self) -> &str {
        "mock/test"
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        // TODO: implement deterministic MD5-based vectors
        todo!("MockEmbeddingProvider::embed not implemented")
    }
}

/// OpenAI embedding provider.
pub struct OpenAiEmbeddingProvider {
    _model: String,
    _dimension: usize,
    _batch_size: usize,
    _client: reqwest::Client,
    _api_key: String,
}

impl OpenAiEmbeddingProvider {
    pub fn new(
        _model: impl Into<String>,
        _dimension: usize,
        _batch_size: usize,
    ) -> Result<Self, EmbeddingError> {
        // TODO: implement
        todo!("OpenAiEmbeddingProvider::new not implemented")
    }
}

/// Google Gemini embedding provider.
pub struct GoogleEmbeddingProvider {
    _model: String,
    _dimension: usize,
    _batch_size: usize,
    _client: reqwest::Client,
    _api_key: String,
}

impl GoogleEmbeddingProvider {
    pub fn new(
        _model: impl Into<String>,
        _dimension: usize,
        _batch_size: usize,
    ) -> Result<Self, EmbeddingError> {
        // TODO: implement
        todo!("GoogleEmbeddingProvider::new not implemented")
    }
}

/// Factory config for creating providers.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmbeddingProviderConfig {
    OpenAi {
        model: String,
        dimension: usize,
        batch_size: Option<usize>,
    },
    Google {
        model: String,
        dimension: usize,
        batch_size: Option<usize>,
    },
    Mock {
        dimension: usize,
    },
}

/// Create provider from config.
pub fn create_embedding_provider(
    _config: &EmbeddingProviderConfig,
) -> Result<Box<dyn EmbeddingProvider>, EmbeddingError> {
    // TODO: implement
    todo!("create_embedding_provider not implemented")
}

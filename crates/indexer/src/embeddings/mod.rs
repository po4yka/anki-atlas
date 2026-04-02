use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

mod deterministic;
mod google;
mod openai;

pub use deterministic::DeterministicEmbeddingProvider;
pub use google::GoogleEmbeddingProvider;
pub use openai::OpenAiEmbeddingProvider;

/// Errors from embedding operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("provider not configured: {0}")]
    NotConfigured(String),
    #[error("http {status}: {body}")]
    Http { status: u16, body: String },
    #[error("retry attempts exhausted after {attempts} tries (last http {status}: {body})")]
    RetryExhausted {
        attempts: u32,
        status: u16,
        body: String,
    },
    #[error("batch embedding failed: {source}")]
    BatchFailed {
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    #[error("rate limited, retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },
    #[error("unsupported input for provider: {message}")]
    UnsupportedInput { message: String },
    #[error("protocol error: {message}")]
    Protocol { message: String },
}

/// Task type used to tune embeddings for downstream retrieval.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingTask {
    #[default]
    Default,
    RetrievalDocument,
    RetrievalQuery,
}

impl EmbeddingTask {
    pub(super) fn google_task_type(self) -> Option<&'static str> {
        match self {
            Self::Default => None,
            Self::RetrievalDocument => Some("RETRIEVAL_DOCUMENT"),
            Self::RetrievalQuery => Some("RETRIEVAL_QUERY"),
        }
    }
}

/// A single multimodal part in an embedding request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EmbeddingPart {
    Text {
        text: String,
    },
    InlineBytes {
        mime_type: String,
        data: Vec<u8>,
        display_name: Option<String>,
    },
    FileUri {
        mime_type: String,
        uri: String,
        display_name: Option<String>,
    },
}

impl EmbeddingPart {
    pub(super) fn append_hash_bytes(&self, hasher: &mut Sha256) {
        match self {
            Self::Text { text } => {
                hasher.update(b"text");
                hasher.update(text.as_bytes());
            }
            Self::InlineBytes {
                mime_type,
                data,
                display_name,
            } => {
                hasher.update(b"inline_bytes");
                hasher.update(mime_type.as_bytes());
                hasher.update(display_name.as_deref().unwrap_or_default().as_bytes());
                hasher.update(data);
            }
            Self::FileUri {
                mime_type,
                uri,
                display_name,
            } => {
                hasher.update(b"file_uri");
                hasher.update(mime_type.as_bytes());
                hasher.update(uri.as_bytes());
                hasher.update(display_name.as_deref().unwrap_or_default().as_bytes());
            }
        }
    }
}

/// A single embedding request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingInput {
    pub parts: Vec<EmbeddingPart>,
    #[serde(default)]
    pub task: EmbeddingTask,
    pub title: Option<String>,
    pub output_dimensionality: Option<usize>,
}

impl EmbeddingInput {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            parts: vec![EmbeddingPart::Text { text: text.into() }],
            task: EmbeddingTask::Default,
            title: None,
            output_dimensionality: None,
        }
    }

    pub fn text_with_task(text: impl Into<String>, task: EmbeddingTask) -> Self {
        Self {
            task,
            ..Self::text(text)
        }
    }

    pub fn with_output_dimensionality(mut self, output_dimensionality: usize) -> Self {
        self.output_dimensionality = Some(output_dimensionality);
        self
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub(super) fn as_single_text(&self) -> Option<&str> {
        match self.parts.as_slice() {
            [EmbeddingPart::Text { text }] => Some(text),
            _ => None,
        }
    }
}

/// Trait for embedding providers. All impls must be Send + Sync.
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait EmbeddingProvider: Send + Sync {
    /// Model identifier for version tracking.
    fn model_name(&self) -> &str;

    /// Dimensionality of output vectors.
    fn dimension(&self) -> usize;

    /// Embed a batch of multimodal inputs. Returns one vector per input.
    async fn embed_inputs(
        &self,
        inputs: &[EmbeddingInput],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Embed a batch of texts. Returns one vector per input text.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let inputs: Vec<_> = texts.iter().cloned().map(EmbeddingInput::text).collect();
        self.embed_inputs(&inputs).await
    }
}

#[async_trait]
impl<T> EmbeddingProvider for &T
where
    T: EmbeddingProvider + ?Sized,
{
    fn model_name(&self) -> &str {
        (*self).model_name()
    }

    fn dimension(&self) -> usize {
        (*self).dimension()
    }

    async fn embed_inputs(
        &self,
        inputs: &[EmbeddingInput],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        (*self).embed_inputs(inputs).await
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        (*self).embed(texts).await
    }
}

#[async_trait]
impl<T> EmbeddingProvider for Box<T>
where
    T: EmbeddingProvider + ?Sized,
{
    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    fn dimension(&self) -> usize {
        (**self).dimension()
    }

    async fn embed_inputs(
        &self,
        inputs: &[EmbeddingInput],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        (**self).embed_inputs(inputs).await
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        (**self).embed(texts).await
    }
}

#[async_trait]
impl<T> EmbeddingProvider for Arc<T>
where
    T: EmbeddingProvider + ?Sized,
{
    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    fn dimension(&self) -> usize {
        (**self).dimension()
    }

    async fn embed_inputs(
        &self,
        inputs: &[EmbeddingInput],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        (**self).embed_inputs(inputs).await
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        (**self).embed(texts).await
    }
}

/// SHA-256[:16] hash of "{model_name}:{text}" for change detection.
pub fn content_hash(model_name: &str, text: &str) -> String {
    content_hash_parts(model_name, [text])
}

/// SHA-256[:16] hash of the model name and all content parts for change detection.
pub fn content_hash_parts<'a, I>(model_name: &str, parts: I) -> String
where
    I: IntoIterator<Item = &'a str>,
{
    let mut hasher = Sha256::new();
    hasher.update(model_name.as_bytes());
    for part in parts {
        hasher.update([0]);
        hasher.update(part.as_bytes());
    }
    let hash = hasher.finalize();
    hex::encode(&hash[..8]) // 16 hex chars = 8 bytes
}

/// Factory config for creating providers.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmbeddingProviderConfig {
    OpenAi {
        model: String,
        dimension: usize,
        batch_size: Option<usize>,
        api_key: String,
    },
    Google {
        model: String,
        dimension: usize,
        batch_size: Option<usize>,
        api_key: String,
    },
    Mock {
        dimension: usize,
    },
}

/// Create provider from config.
pub fn create_embedding_provider(
    config: &EmbeddingProviderConfig,
) -> Result<Box<dyn EmbeddingProvider>, EmbeddingError> {
    match config {
        EmbeddingProviderConfig::Mock { dimension } => {
            Ok(Box::new(DeterministicEmbeddingProvider::new(*dimension)))
        }
        EmbeddingProviderConfig::OpenAi {
            model,
            dimension,
            batch_size,
            api_key,
        } => OpenAiEmbeddingProvider::new(
            model.clone(),
            *dimension,
            batch_size.unwrap_or(100),
            api_key.clone(),
        )
        .map(|p| Box::new(p) as Box<dyn EmbeddingProvider>),
        EmbeddingProviderConfig::Google {
            model,
            dimension,
            batch_size,
            api_key,
        } => GoogleEmbeddingProvider::new(
            model.clone(),
            *dimension,
            batch_size.unwrap_or(100),
            api_key.clone(),
        )
        .map(|p| Box::new(p) as Box<dyn EmbeddingProvider>),
    }
}

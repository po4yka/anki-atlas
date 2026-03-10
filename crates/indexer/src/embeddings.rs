use async_trait::async_trait;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::sync::Arc;

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

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        (**self).embed(texts).await
    }
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
    dimension: usize,
}

impl MockEmbeddingProvider {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    fn model_name(&self) -> &str {
        "mock/test"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            let hash_bytes = Sha256::digest(text.as_bytes());
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

/// OpenAI embedding provider.
pub struct OpenAiEmbeddingProvider {
    model: String,
    dimension: usize,
    batch_size: usize,
    client: reqwest::Client,
    api_key: String,
}

const MAX_RETRIES: u32 = 5;
const RETRYABLE_STATUS_CODES: &[u16] = &[429, 502, 503, 504];

impl OpenAiEmbeddingProvider {
    pub fn new(
        model: impl Into<String>,
        dimension: usize,
        batch_size: usize,
        api_key: impl Into<String>,
    ) -> Result<Self, EmbeddingError> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(EmbeddingError::NotConfigured(
                "OpenAI api_key must be provided".into(),
            ));
        }
        Ok(Self {
            model: model.into(),
            dimension,
            batch_size,
            client: reqwest::Client::new(),
            api_key,
        })
    }
}

#[derive(Deserialize)]
struct OpenAiEmbeddingItem {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct OpenAiEmbeddingResponse {
    data: Vec<OpenAiEmbeddingItem>,
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbeddingProvider {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for batch in texts.chunks(self.batch_size) {
            let body = serde_json::json!({
                "model": self.model,
                "input": batch,
            });

            let mut last_err = None;
            for attempt in 0..MAX_RETRIES {
                let response = self
                    .client
                    .post("https://api.openai.com/v1/embeddings")
                    .bearer_auth(&self.api_key)
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| EmbeddingError::BatchFailed {
                        source: Box::new(e),
                    })?;

                let status = response.status().as_u16();

                if status == 200 {
                    let parsed: OpenAiEmbeddingResponse =
                        response
                            .json()
                            .await
                            .map_err(|e| EmbeddingError::BatchFailed {
                                source: Box::new(e),
                            })?;
                    let mut items = parsed.data;
                    items.sort_by_key(|item| item.index);
                    all_embeddings.extend(items.into_iter().map(|item| item.embedding));
                    last_err = None;
                    break;
                }

                let body_text = response.text().await.unwrap_or_default();

                if !RETRYABLE_STATUS_CODES.contains(&status) {
                    return Err(EmbeddingError::Http {
                        status,
                        body: body_text,
                    });
                }

                last_err = Some((status, body_text));
                if attempt + 1 < MAX_RETRIES {
                    let delay = std::time::Duration::from_secs(1 << (attempt + 1));
                    tokio::time::sleep(delay).await;
                }
            }

            if let Some((status, body)) = last_err {
                return Err(EmbeddingError::RetryExhausted {
                    attempts: MAX_RETRIES,
                    status,
                    body,
                });
            }
        }

        Ok(all_embeddings)
    }
}

/// Google Gemini embedding provider.
pub struct GoogleEmbeddingProvider {
    model: String,
    dimension: usize,
    batch_size: usize,
    client: reqwest::Client,
    api_key: String,
}

impl GoogleEmbeddingProvider {
    pub fn new(
        model: impl Into<String>,
        dimension: usize,
        batch_size: usize,
        api_key: impl Into<String>,
    ) -> Result<Self, EmbeddingError> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(EmbeddingError::NotConfigured(
                "Google api_key must be provided".into(),
            ));
        }
        Ok(Self {
            model: model.into(),
            dimension,
            batch_size,
            client: reqwest::Client::new(),
            api_key,
        })
    }
}

#[derive(Deserialize)]
struct GoogleEmbeddingValue {
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct GoogleBatchEmbedResponse {
    embeddings: Vec<GoogleEmbeddingValue>,
}

#[async_trait]
impl EmbeddingProvider for GoogleEmbeddingProvider {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for (batch_idx, batch) in texts.chunks(self.batch_size).enumerate() {
            if batch_idx > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }

            let requests: Vec<serde_json::Value> = batch
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "model": format!("models/{}", self.model),
                        "content": { "parts": [{ "text": t }] }
                    })
                })
                .collect();

            let body = serde_json::json!({ "requests": requests });
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents?key={}",
                self.model, self.api_key
            );

            let mut last_err = None;
            for attempt in 0..MAX_RETRIES {
                let response = self
                    .client
                    .post(&url)
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| EmbeddingError::BatchFailed {
                        source: Box::new(e),
                    })?;

                let status = response.status().as_u16();

                if status == 200 {
                    let parsed: GoogleBatchEmbedResponse =
                        response
                            .json()
                            .await
                            .map_err(|e| EmbeddingError::BatchFailed {
                                source: Box::new(e),
                            })?;
                    all_embeddings.extend(parsed.embeddings.into_iter().map(|e| e.values));
                    last_err = None;
                    break;
                }

                let body_text = response.text().await.unwrap_or_default();

                if !RETRYABLE_STATUS_CODES.contains(&status) {
                    return Err(EmbeddingError::Http {
                        status,
                        body: body_text,
                    });
                }

                last_err = Some((status, body_text));
                if attempt + 1 < MAX_RETRIES {
                    let delay = std::time::Duration::from_secs(1 << (attempt + 1));
                    tokio::time::sleep(delay).await;
                }
            }

            if let Some((status, body)) = last_err {
                return Err(EmbeddingError::RetryExhausted {
                    attempts: MAX_RETRIES,
                    status,
                    body,
                });
            }
        }

        Ok(all_embeddings)
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
            Ok(Box::new(MockEmbeddingProvider::new(*dimension)))
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

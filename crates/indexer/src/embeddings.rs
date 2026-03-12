use async_trait::async_trait;
use base64::Engine as _;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
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
    fn google_task_type(self) -> Option<&'static str> {
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
    fn append_hash_bytes(&self, hasher: &mut Sha256) {
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

    fn as_single_text(&self) -> Option<&str> {
        match self.parts.as_slice() {
            [EmbeddingPart::Text { text }] => Some(text),
            _ => None,
        }
    }
}

/// Trait for embedding providers. All impls must be Send + Sync.
#[async_trait]
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

impl OpenAiEmbeddingProvider {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for batch in texts.chunks(self.batch_size) {
            let body = serde_json::json!({
                "model": self.model,
                "input": batch,
                "dimensions": self.dimension,
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

#[async_trait]
impl EmbeddingProvider for OpenAiEmbeddingProvider {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed_inputs(
        &self,
        inputs: &[EmbeddingInput],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let texts = inputs
            .iter()
            .map(|input| {
                input.as_single_text().map(str::to_string).ok_or_else(|| {
                    EmbeddingError::UnsupportedInput {
                        message: "OpenAI embeddings only support single text parts".to_string(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        for input in inputs {
            if let Some(output_dimensionality) = input.output_dimensionality
                && output_dimensionality != self.dimension
            {
                return Err(EmbeddingError::UnsupportedInput {
                    message: format!(
                        "OpenAI provider is configured for {} dimensions but received {}",
                        self.dimension, output_dimensionality
                    ),
                });
            }
        }

        self.embed_texts(&texts).await
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

const GEMINI_EMBEDDING_2_MODEL: &str = "gemini-embedding-2-preview";
const GOOGLE_INLINE_BYTES_LIMIT: usize = 5 * 1024 * 1024;

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

    fn uses_multimodal_path(&self) -> bool {
        self.model == GEMINI_EMBEDDING_2_MODEL
    }

    fn build_legacy_batch_body(
        &self,
        texts: &[String],
    ) -> Result<serde_json::Value, EmbeddingError> {
        if self.uses_multimodal_path() {
            return Err(EmbeddingError::UnsupportedInput {
                message: "legacy batch body is not used for Gemini Embedding 2".to_string(),
            });
        }
        let requests: Vec<serde_json::Value> = texts
            .iter()
            .map(|text| {
                serde_json::json!({
                    "model": format!("models/{}", self.model),
                    "content": { "parts": [{ "text": text }] }
                })
            })
            .collect();

        Ok(serde_json::json!({ "requests": requests }))
    }

    async fn embed_legacy_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for (batch_idx, batch) in texts.chunks(self.batch_size).enumerate() {
            if batch_idx > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }

            let body = self.build_legacy_batch_body(batch)?;
            let url = format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:batchEmbedContents",
                self.model
            );

            let mut last_err = None;
            for attempt in 0..MAX_RETRIES {
                let response = self
                    .client
                    .post(&url)
                    .header("x-goog-api-key", &self.api_key)
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

    async fn build_multimodal_parts(
        &self,
        parts: &[EmbeddingPart],
    ) -> Result<Vec<serde_json::Value>, EmbeddingError> {
        let mut payload_parts = Vec::with_capacity(parts.len());
        for part in parts {
            match part {
                EmbeddingPart::Text { text } => {
                    payload_parts.push(serde_json::json!({ "text": text }));
                }
                EmbeddingPart::InlineBytes {
                    mime_type,
                    data,
                    display_name,
                } => {
                    if data.len() > GOOGLE_INLINE_BYTES_LIMIT {
                        let file_uri = self
                            .upload_file_bytes(mime_type, data, display_name.as_deref())
                            .await?;
                        payload_parts.push(serde_json::json!({
                            "file_data": {
                                "mime_type": mime_type,
                                "file_uri": file_uri,
                            }
                        }));
                    } else {
                        payload_parts.push(serde_json::json!({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64::engine::general_purpose::STANDARD.encode(data),
                            }
                        }));
                    }
                }
                EmbeddingPart::FileUri {
                    mime_type,
                    uri,
                    display_name: _,
                } => {
                    payload_parts.push(serde_json::json!({
                        "file_data": {
                            "mime_type": mime_type,
                            "file_uri": uri,
                        }
                    }));
                }
            }
        }
        Ok(payload_parts)
    }

    async fn build_multimodal_body(
        &self,
        input: &EmbeddingInput,
    ) -> Result<serde_json::Value, EmbeddingError> {
        let mut body = serde_json::json!({
            "content": {
                "parts": self.build_multimodal_parts(&input.parts).await?,
            },
            "outputDimensionality": input.output_dimensionality.unwrap_or(self.dimension),
        });
        if let Some(task_type) = input.task.google_task_type() {
            body["taskType"] = serde_json::Value::String(task_type.to_string());
        }
        if let Some(title) = input.title.clone() {
            body["title"] = serde_json::Value::String(title);
        }
        Ok(body)
    }

    async fn upload_file_bytes(
        &self,
        mime_type: &str,
        data: &[u8],
        display_name: Option<&str>,
    ) -> Result<String, EmbeddingError> {
        let metadata = serde_json::json!({
            "file": {
                "display_name": display_name.unwrap_or("anki-atlas-asset"),
            }
        });
        let start_response = self
            .client
            .post("https://generativelanguage.googleapis.com/upload/v1beta/files")
            .header("x-goog-api-key", &self.api_key)
            .header("X-Goog-Upload-Protocol", "resumable")
            .header("X-Goog-Upload-Command", "start")
            .header(
                "X-Goog-Upload-Header-Content-Length",
                data.len().to_string(),
            )
            .header("X-Goog-Upload-Header-Content-Type", mime_type)
            .header("Content-Type", "application/json")
            .json(&metadata)
            .send()
            .await
            .map_err(|error| EmbeddingError::BatchFailed {
                source: Box::new(error),
            })?;

        let status = start_response.status().as_u16();
        if status != 200 && status != 201 {
            return Err(EmbeddingError::Http {
                status,
                body: start_response.text().await.unwrap_or_default(),
            });
        }

        let upload_url = start_response
            .headers()
            .get("x-goog-upload-url")
            .or_else(|| start_response.headers().get("X-Goog-Upload-URL"))
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
            .ok_or_else(|| EmbeddingError::Protocol {
                message: "Files API did not return x-goog-upload-url".to_string(),
            })?;

        let upload_response = self
            .client
            .post(upload_url)
            .header("Content-Length", data.len().to_string())
            .header("X-Goog-Upload-Offset", "0")
            .header("X-Goog-Upload-Command", "upload, finalize")
            .body(data.to_vec())
            .send()
            .await
            .map_err(|error| EmbeddingError::BatchFailed {
                source: Box::new(error),
            })?;

        let status = upload_response.status().as_u16();
        if status != 200 && status != 201 {
            return Err(EmbeddingError::Http {
                status,
                body: upload_response.text().await.unwrap_or_default(),
            });
        }

        let uploaded: GoogleFileUploadResponse =
            upload_response
                .json()
                .await
                .map_err(|error| EmbeddingError::BatchFailed {
                    source: Box::new(error),
                })?;
        uploaded.file.uri.ok_or_else(|| EmbeddingError::Protocol {
            message: "Files API upload response did not include file.uri".to_string(),
        })
    }

    async fn embed_single_multimodal(
        &self,
        input: &EmbeddingInput,
    ) -> Result<Vec<f32>, EmbeddingError> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent",
            self.model
        );
        let body = self.build_multimodal_body(input).await?;

        let mut last_err = None;
        for attempt in 0..MAX_RETRIES {
            let response = self
                .client
                .post(&url)
                .header("x-goog-api-key", &self.api_key)
                .json(&body)
                .send()
                .await
                .map_err(|error| EmbeddingError::BatchFailed {
                    source: Box::new(error),
                })?;

            let status = response.status().as_u16();
            if status == 200 {
                let parsed: GoogleEmbedContentResponse =
                    response
                        .json()
                        .await
                        .map_err(|error| EmbeddingError::BatchFailed {
                            source: Box::new(error),
                        })?;
                return Ok(parsed.embedding.values);
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

        Err(EmbeddingError::Protocol {
            message: "multimodal embedding request completed without response".to_string(),
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

#[derive(Deserialize)]
struct GoogleEmbedContentResponse {
    embedding: GoogleEmbeddingValue,
}

#[derive(Debug, Deserialize)]
struct GoogleFileUploadResponse {
    file: GoogleUploadedFile,
}

#[derive(Debug, Deserialize)]
struct GoogleUploadedFile {
    uri: Option<String>,
}

#[async_trait]
impl EmbeddingProvider for GoogleEmbeddingProvider {
    fn model_name(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed_inputs(
        &self,
        inputs: &[EmbeddingInput],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        if !self.uses_multimodal_path() {
            let texts = inputs
                .iter()
                .map(|input| {
                    if input.task != EmbeddingTask::Default
                        || input.title.is_some()
                        || input.output_dimensionality.is_some()
                    {
                        return Err(EmbeddingError::UnsupportedInput {
                            message: format!(
                                "{} only supports plain text batch embedding",
                                self.model
                            ),
                        });
                    }
                    input.as_single_text().map(str::to_string).ok_or_else(|| {
                        EmbeddingError::UnsupportedInput {
                            message: format!("{} only supports single text parts", self.model),
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            return self.embed_legacy_texts(&texts).await;
        }

        let concurrency = self.batch_size.clamp(1, 8);
        futures::stream::iter(
            inputs
                .iter()
                .cloned()
                .map(|input| async move { self.embed_single_multimodal(&input).await }),
        )
        .buffered(concurrency)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect()
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

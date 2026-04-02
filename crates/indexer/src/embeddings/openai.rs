use async_trait::async_trait;
use serde::Deserialize;

use super::{EmbeddingError, EmbeddingInput, EmbeddingProvider};

const MAX_RETRIES: u32 = 5;
const RETRYABLE_STATUS_CODES: &[u16] = &[429, 502, 503, 504];

/// OpenAI embedding provider.
pub struct OpenAiEmbeddingProvider {
    model: String,
    dimension: usize,
    batch_size: usize,
    client: reqwest::Client,
    api_key: String,
}

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

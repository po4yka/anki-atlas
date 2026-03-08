use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use tracing::instrument;

use crate::error::LlmError;
use crate::provider::{GenerateOptions, LlmProvider};
use crate::response::LlmResponse;

/// OpenRouter provider configuration.
#[derive(Debug, Clone)]
pub struct OpenRouterConfig {
    pub api_key: String,
    pub base_url: String,
    pub timeout_secs: u64,
    pub max_tokens: u32,
    pub max_retries: u32,
    pub site_url: Option<String>,
    pub site_name: Option<String>,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            timeout_secs: 180,
            max_tokens: 2048,
            max_retries: 3,
            site_url: None,
            site_name: None,
        }
    }
}

/// OpenRouter LLM provider with retry logic and exponential backoff.
pub struct OpenRouterProvider {
    config: OpenRouterConfig,
    client: Client,
}

const RETRYABLE_STATUS_CODES: &[u16] = &[429, 502, 503, 504];

impl OpenRouterProvider {
    /// Create a new provider. Returns LlmError::Provider if api_key is empty.
    pub fn new(config: OpenRouterConfig) -> Result<Self, LlmError> {
        if config.api_key.is_empty() {
            return Err(LlmError::Provider {
                message: "API key must not be empty".to_string(),
                source: None,
            });
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| LlmError::Provider {
                message: format!("failed to build HTTP client: {e}"),
                source: Some(Box::new(e)),
            })?;

        Ok(Self { config, client })
    }

    fn build_messages(&self, prompt: &str, opts: &GenerateOptions) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();
        if !opts.system.is_empty() {
            messages.push(json!({"role": "system", "content": opts.system}));
        }
        messages.push(json!({"role": "user", "content": prompt}));
        messages
    }

    fn build_request_body(
        &self,
        model: &str,
        prompt: &str,
        opts: &GenerateOptions,
    ) -> serde_json::Value {
        let mut body = json!({
            "model": model,
            "messages": self.build_messages(prompt, opts),
            "max_tokens": self.config.max_tokens,
            "temperature": opts.temperature,
        });

        if let Some(schema) = &opts.json_schema {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": schema,
            });
        } else if opts.json_mode {
            body["response_format"] = json!({"type": "json_object"});
        }

        body
    }

    fn parse_response(
        &self,
        model: &str,
        body: &serde_json::Value,
    ) -> Result<LlmResponse, LlmError> {
        let choices = body["choices"]
            .as_array()
            .ok_or_else(|| LlmError::Provider {
                message: "response missing choices array".to_string(),
                source: None,
            })?;

        if choices.is_empty() {
            return Err(LlmError::Provider {
                message: "response has empty choices array".to_string(),
                source: None,
            });
        }

        let choice = &choices[0];
        let text = choice["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string();

        let response_model = body["model"]
            .as_str()
            .map(String::from)
            .unwrap_or_else(|| model.to_string());

        let prompt_tokens = body["usage"]["prompt_tokens"].as_u64().map(|v| v as u32);
        let completion_tokens = body["usage"]["completion_tokens"]
            .as_u64()
            .map(|v| v as u32);
        let finish_reason = choice["finish_reason"].as_str().map(String::from);

        let raw: HashMap<String, serde_json::Value> =
            serde_json::from_value(body.clone()).unwrap_or_default();

        Ok(LlmResponse {
            text,
            model: response_model,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            raw,
        })
    }
}

#[async_trait]
impl LlmProvider for OpenRouterProvider {
    #[instrument(skip(self, prompt, opts))]
    async fn generate(
        &self,
        model: &str,
        prompt: &str,
        opts: &GenerateOptions,
    ) -> Result<LlmResponse, LlmError> {
        let url = format!("{}/chat/completions", self.config.base_url);
        let request_body = self.build_request_body(model, prompt, opts);

        for attempt in 0..self.config.max_retries {
            let mut req = self
                .client
                .post(&url)
                .bearer_auth(&self.config.api_key)
                .json(&request_body);

            if let Some(site_url) = &self.config.site_url {
                req = req.header("HTTP-Referer", site_url);
            }
            if let Some(site_name) = &self.config.site_name {
                req = req.header("X-Title", site_name);
            }

            let response = req.send().await.map_err(|e| LlmError::Connection(e.to_string()))?;
            let status = response.status().as_u16();

            if status == 200 {
                let body: serde_json::Value =
                    response.json().await.map_err(|e| LlmError::Provider {
                        message: format!("failed to parse response body: {e}"),
                        source: Some(Box::new(e)),
                    })?;
                return self.parse_response(model, &body);
            }

            if !RETRYABLE_STATUS_CODES.contains(&status) {
                let body = response.text().await.unwrap_or_default();
                return Err(LlmError::Http { status, body });
            }

            // Retryable status - sleep with exponential backoff if not last attempt
            if attempt + 1 < self.config.max_retries {
                let delay = Duration::from_secs(1 << (attempt + 1));
                tokio::time::sleep(delay).await;
            }
        }

        Err(LlmError::Exhausted {
            message: "all retries exhausted".to_string(),
            attempts: self.config.max_retries,
        })
    }

    #[instrument(skip(self))]
    async fn check_connection(&self) -> bool {
        let url = format!("{}/models", self.config.base_url);
        self.client
            .get(&url)
            .bearer_auth(&self.config.api_key)
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
    }

    #[instrument(skip(self))]
    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let url = format!("{}/models", self.config.base_url);
        let response = self
            .client
            .get(&url)
            .bearer_auth(&self.config.api_key)
            .send()
            .await
            .map_err(|e| LlmError::Connection(e.to_string()))?;

        let body: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LlmError::Provider {
                message: format!("failed to parse models response: {e}"),
                source: Some(Box::new(e)),
            })?;

        let models = body["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }
}

#[cfg(test)]
mod tests;

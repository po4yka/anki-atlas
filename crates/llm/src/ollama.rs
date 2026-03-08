use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use tracing::instrument;

use crate::error::LlmError;
use crate::provider::{GenerateOptions, LlmProvider};
use crate::response::LlmResponse;

/// Ollama provider configuration.
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub timeout_secs: u64,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            api_key: None,
            timeout_secs: 300,
        }
    }
}

/// Ollama LLM provider for local and cloud deployments.
pub struct OllamaProvider {
    config: OllamaConfig,
    client: Client,
}

impl OllamaProvider {
    /// Create a new Ollama provider with the given configuration.
    pub fn new(config: OllamaConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("failed to build HTTP client");
        Self { config, client }
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    #[instrument(skip(self, prompt, opts))]
    async fn generate(
        &self,
        model: &str,
        prompt: &str,
        opts: &GenerateOptions,
    ) -> Result<LlmResponse, LlmError> {
        let mut body = json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
            "options": {
                "temperature": opts.temperature
            }
        });

        if !opts.system.is_empty() {
            body["system"] = json!(opts.system);
        }

        if opts.json_mode {
            body["format"] = json!("json");
        }

        let mut req = self
            .client
            .post(format!("{}/api/generate", self.config.base_url))
            .json(&body);

        if let Some(ref api_key) = self.config.api_key {
            req = req.header("Authorization", format!("Bearer {api_key}"));
        }

        let resp = req
            .send()
            .await
            .map_err(|e| LlmError::Connection(e.to_string()))?;

        let status = resp.status().as_u16();
        if status != 200 {
            let body_text = resp.text().await.unwrap_or_default();
            return Err(LlmError::Http {
                status,
                body: body_text,
            });
        }

        let resp_json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Connection(e.to_string()))?;

        let text = resp_json["response"]
            .as_str()
            .unwrap_or_default()
            .to_string();

        let resp_model = resp_json["model"]
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_else(|| model.to_string());

        let prompt_tokens = resp_json["prompt_eval_count"].as_u64().map(|n| n as u32);
        let completion_tokens = resp_json["eval_count"].as_u64().map(|n| n as u32);
        let finish_reason = resp_json["done_reason"].as_str().map(|s| s.to_string());

        let raw: HashMap<String, serde_json::Value> =
            serde_json::from_value(resp_json).unwrap_or_default();

        Ok(LlmResponse {
            text,
            model: resp_model,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            raw,
        })
    }

    #[instrument(skip(self))]
    async fn check_connection(&self) -> bool {
        let result = self
            .client
            .get(format!("{}/api/tags", self.config.base_url))
            .send()
            .await;
        matches!(result, Ok(resp) if resp.status().is_success())
    }

    #[instrument(skip(self))]
    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        let resp = self
            .client
            .get(format!("{}/api/tags", self.config.base_url))
            .send()
            .await
            .map_err(|e| LlmError::Connection(e.to_string()))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Connection(e.to_string()))?;

        let models = body["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["name"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }
}

#[cfg(test)]
mod tests;

use async_trait::async_trait;
use reqwest::Client;

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

#[allow(dead_code)] // Fields used once new() is implemented (GREEN phase)
pub struct OpenRouterProvider {
    config: OpenRouterConfig,
    client: Client,
}

impl OpenRouterProvider {
    /// Create a new provider. Returns LlmError::Provider if api_key is empty.
    pub fn new(_config: OpenRouterConfig) -> Result<Self, LlmError> {
        todo!()
    }
}

#[async_trait]
impl LlmProvider for OpenRouterProvider {
    async fn generate(
        &self,
        _model: &str,
        _prompt: &str,
        _opts: &GenerateOptions,
    ) -> Result<LlmResponse, LlmError> {
        todo!()
    }

    async fn check_connection(&self) -> bool {
        todo!()
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        todo!()
    }
}

#[cfg(test)]
mod tests;

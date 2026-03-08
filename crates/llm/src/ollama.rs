#![allow(unused_imports)]

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
#[allow(dead_code)]
pub struct OllamaProvider {
    config: OllamaConfig,
    client: Client,
}

impl OllamaProvider {
    #[allow(unused_variables)]
    pub fn new(config: OllamaConfig) -> Self {
        todo!()
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    #[instrument(skip(self, prompt, opts))]
    #[allow(unused_variables)]
    async fn generate(
        &self,
        model: &str,
        prompt: &str,
        opts: &GenerateOptions,
    ) -> Result<LlmResponse, LlmError> {
        todo!()
    }

    #[instrument(skip(self))]
    async fn check_connection(&self) -> bool {
        todo!()
    }

    #[instrument(skip(self))]
    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        todo!()
    }
}

#[cfg(test)]
mod tests;

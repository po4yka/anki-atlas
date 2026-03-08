use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::LlmError;
use crate::response::LlmResponse;

/// Options for a generation request.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub system: String,
    pub temperature: f32,
    pub json_mode: bool,
    pub json_schema: Option<serde_json::Value>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            system: String::new(),
            temperature: 0.7,
            json_mode: false,
            json_schema: None,
        }
    }
}

/// Trait for LLM providers. All implementations must be Send + Sync.
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait LlmProvider: Send + Sync {
    /// Generate a completion.
    async fn generate(
        &self,
        model: &str,
        prompt: &str,
        opts: &GenerateOptions,
    ) -> Result<LlmResponse, LlmError>;

    /// Generate and parse JSON output.
    async fn generate_json(
        &self,
        model: &str,
        prompt: &str,
        opts: &GenerateOptions,
    ) -> Result<HashMap<String, serde_json::Value>, LlmError> {
        let mut json_opts = opts.clone();
        json_opts.json_mode = true;
        let response = self.generate(model, prompt, &json_opts).await?;
        serde_json::from_str(&response.text).map_err(|e| LlmError::InvalidJson {
            message: format!("LLM returned invalid JSON: {e}"),
            response_text: response.text[..response.text.len().min(500)].to_string(),
        })
    }

    /// Check if the provider API is reachable.
    async fn check_connection(&self) -> bool;

    /// List available model identifiers.
    async fn list_models(&self) -> Result<Vec<String>, LlmError>;
}

#[cfg(test)]
mod tests;

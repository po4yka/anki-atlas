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
        let _ = (model, prompt, opts);
        todo!("generate_json default implementation")
    }

    /// Check if the provider API is reachable.
    async fn check_connection(&self) -> bool;

    /// List available model identifiers.
    async fn list_models(&self) -> Result<Vec<String>, LlmError>;
}

#[cfg(test)]
mod tests;

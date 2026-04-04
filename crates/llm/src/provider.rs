use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::LlmError;
use crate::response::LlmResponse;

/// A single part of a multimodal message content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text content.
    Text { text: String },
    /// Image referenced by URL.
    ImageUrl { url: String },
    /// Base64-encoded image data.
    ImageBase64 { mime_type: String, data: String },
}

/// Options for a generation request.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub system: String,
    pub temperature: f32,
    pub json_mode: bool,
    pub json_schema: Option<serde_json::Value>,
    /// Additional multimodal content parts (images, etc.).
    /// Appended after the text prompt in the message.
    /// Empty by default for text-only generation.
    pub content_parts: Vec<ContentPart>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            system: String::new(),
            temperature: 0.7,
            json_mode: false,
            json_schema: None,
            content_parts: Vec::new(),
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

    /// Check if the provider API is reachable.
    async fn check_connection(&self) -> bool;

    /// List available model identifiers.
    async fn list_models(&self) -> Result<Vec<String>, LlmError>;
}

#[cfg(test)]
mod tests;

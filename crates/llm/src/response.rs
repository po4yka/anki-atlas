use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Structured response from an LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub text: String,
    pub model: String,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub finish_reason: Option<String>,
    pub raw: HashMap<String, serde_json::Value>,
}

impl LlmResponse {
    /// Total tokens used (prompt + completion), if both are known.
    pub fn total_tokens(&self) -> Option<u32> {
        match (self.prompt_tokens, self.completion_tokens) {
            (Some(p), Some(c)) => Some(p + c),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;

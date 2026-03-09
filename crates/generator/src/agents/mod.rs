pub mod llm_enhancer;
pub mod llm_generator;
pub mod llm_validator;

use std::sync::Arc;

use async_trait::async_trait;

use llm::{GenerateOptions, LlmProvider};

use crate::error::GeneratorError;
use crate::models::*;

#[cfg(test)]
mod tests;

/// Common base for LLM-backed agents. Holds the provider, model, and temperature.
pub(crate) struct LlmAgentBase {
    pub provider: Arc<dyn LlmProvider>,
    pub model_name: String,
    pub temperature: f32,
}

impl LlmAgentBase {
    pub fn new(provider: Arc<dyn LlmProvider>, model_name: String, temperature: f32) -> Self {
        Self {
            provider,
            model_name,
            temperature,
        }
    }

    pub fn generate_opts(&self) -> GenerateOptions {
        GenerateOptions {
            temperature: self.temperature,
            json_mode: true,
            ..Default::default()
        }
    }

    pub async fn call_llm(&self, prompt: &str) -> Result<String, GeneratorError> {
        let opts = self.generate_opts();
        let response = self
            .provider
            .generate(&self.model_name, prompt, &opts)
            .await?;
        Ok(response.text)
    }
}

/// Trait for card generation from Q/A pairs.
#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait GeneratorAgent: Send + Sync {
    async fn generate(
        &self,
        deps: &GenerationDeps,
        qa_pairs: &[(String, String)],
    ) -> Result<GenerationResult, GeneratorError>;
}

/// Trait for card enhancement and split suggestions.
#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait EnhancerAgent: Send + Sync {
    /// Enhance a single card.
    async fn enhance(
        &self,
        card: &GeneratedCard,
        deps: &GenerationDeps,
    ) -> Result<GeneratedCard, GeneratorError>;

    /// Analyze content and suggest whether to split into multiple cards.
    async fn suggest_split(
        &self,
        content: &str,
        deps: &GenerationDeps,
    ) -> Result<SplitDecision, GeneratorError>;
}

/// Trait for content validation.
#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait ValidatorAgent: Send + Sync {
    async fn validate(
        &self,
        content: &str,
        deps: &GenerationDeps,
    ) -> Result<ValidationResult, GeneratorError>;
}

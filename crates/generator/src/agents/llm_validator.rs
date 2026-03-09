use std::sync::Arc;

use async_trait::async_trait;
use tracing::instrument;

use llm::LlmProvider;

use crate::agents::{LlmAgentBase, ValidatorAgent};
use crate::error::GeneratorError;
use crate::models::{GenerationDeps, ValidationResult};

/// LLM-backed validator agent parameterized by prompt prefix.
struct LlmValidatorAgent {
    base: LlmAgentBase,
    prompt_prefix: &'static str,
}

impl LlmValidatorAgent {
    fn new(
        provider: Arc<dyn LlmProvider>,
        model_name: String,
        temperature: f32,
        prompt_prefix: &'static str,
    ) -> Self {
        Self {
            base: LlmAgentBase::new(provider, model_name, temperature),
            prompt_prefix,
        }
    }

    async fn validate_impl(
        &self,
        content: &str,
        deps: &GenerationDeps,
    ) -> Result<ValidationResult, GeneratorError> {
        let prompt = format!(
            "{} Note: '{}'. Content: {}",
            self.prompt_prefix, deps.note_title, content
        );

        let text = self.base.call_llm(&prompt).await?;

        serde_json::from_str(&text).map_err(|e| GeneratorError::Validation {
            message: format!("Failed to parse validation result: {e}"),
        })
    }
}

/// LLM-backed pre-validator agent.
pub struct LlmPreValidatorAgent(LlmValidatorAgent);

impl LlmPreValidatorAgent {
    pub fn new(provider: Arc<dyn LlmProvider>, model_name: String, temperature: f32) -> Self {
        Self(LlmValidatorAgent::new(
            provider,
            model_name,
            temperature,
            "Pre-validate content for card generation.",
        ))
    }
}

#[async_trait]
impl ValidatorAgent for LlmPreValidatorAgent {
    #[instrument(skip_all)]
    async fn validate(
        &self,
        content: &str,
        deps: &GenerationDeps,
    ) -> Result<ValidationResult, GeneratorError> {
        self.0.validate_impl(content, deps).await
    }
}

/// LLM-backed post-validator agent.
pub struct LlmPostValidatorAgent(LlmValidatorAgent);

impl LlmPostValidatorAgent {
    pub fn new(provider: Arc<dyn LlmProvider>, model_name: String, temperature: f32) -> Self {
        Self(LlmValidatorAgent::new(
            provider,
            model_name,
            temperature,
            "Post-validate generated card content.",
        ))
    }
}

#[async_trait]
impl ValidatorAgent for LlmPostValidatorAgent {
    #[instrument(skip_all)]
    async fn validate(
        &self,
        content: &str,
        deps: &GenerationDeps,
    ) -> Result<ValidationResult, GeneratorError> {
        self.0.validate_impl(content, deps).await
    }
}

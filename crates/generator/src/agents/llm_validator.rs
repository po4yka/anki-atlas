use std::sync::Arc;

use async_trait::async_trait;
use tracing::instrument;

use llm::{GenerateOptions, LlmProvider};

use crate::agents::ValidatorAgent;
use crate::error::GeneratorError;
use crate::models::{GenerationDeps, ValidationResult};

/// LLM-backed pre-validator agent.
pub struct LlmPreValidatorAgent {
    provider: Arc<dyn LlmProvider>,
    model_name: String,
    temperature: f32,
}

impl LlmPreValidatorAgent {
    pub fn new(provider: Arc<dyn LlmProvider>, model_name: String, temperature: f32) -> Self {
        Self {
            provider,
            model_name,
            temperature,
        }
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
        let prompt = format!(
            "Pre-validate content for card generation. Note: '{}'. Content: {}",
            deps.note_title, content
        );

        let opts = GenerateOptions {
            temperature: self.temperature,
            json_mode: true,
            ..Default::default()
        };

        let response = self
            .provider
            .generate(&self.model_name, &prompt, &opts)
            .await?;

        let result: ValidationResult =
            serde_json::from_str(&response.text).map_err(|e| GeneratorError::Validation {
                message: format!("Failed to parse validation result: {e}"),
            })?;

        Ok(result)
    }
}

/// LLM-backed post-validator agent.
pub struct LlmPostValidatorAgent {
    provider: Arc<dyn LlmProvider>,
    model_name: String,
    temperature: f32,
}

impl LlmPostValidatorAgent {
    pub fn new(provider: Arc<dyn LlmProvider>, model_name: String, temperature: f32) -> Self {
        Self {
            provider,
            model_name,
            temperature,
        }
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
        let prompt = format!(
            "Post-validate generated card content. Note: '{}'. Content: {}",
            deps.note_title, content
        );

        let opts = GenerateOptions {
            temperature: self.temperature,
            json_mode: true,
            ..Default::default()
        };

        let response = self
            .provider
            .generate(&self.model_name, &prompt, &opts)
            .await?;

        let result: ValidationResult =
            serde_json::from_str(&response.text).map_err(|e| GeneratorError::Validation {
                message: format!("Failed to parse validation result: {e}"),
            })?;

        Ok(result)
    }
}

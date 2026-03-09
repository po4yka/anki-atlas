use std::sync::Arc;

use async_trait::async_trait;
use tracing::instrument;

use llm::{GenerateOptions, LlmProvider};

use crate::agents::EnhancerAgent;
use crate::error::GeneratorError;
use crate::models::{GeneratedCard, GenerationDeps, SplitDecision};

/// LLM-backed enhancer agent.
pub struct LlmEnhancerAgent {
    provider: Arc<dyn LlmProvider>,
    model_name: String,
    temperature: f32,
}

impl LlmEnhancerAgent {
    pub fn new(provider: Arc<dyn LlmProvider>, model_name: String, temperature: f32) -> Self {
        Self {
            provider,
            model_name,
            temperature,
        }
    }
}

#[async_trait]
impl EnhancerAgent for LlmEnhancerAgent {
    #[instrument(skip_all)]
    async fn enhance(
        &self,
        card: &GeneratedCard,
        deps: &GenerationDeps,
    ) -> Result<GeneratedCard, GeneratorError> {
        let prompt = format!(
            "Enhance this Anki card for note '{}'. Current HTML: {}",
            deps.note_title, card.apf_html
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

        let json: serde_json::Value =
            serde_json::from_str(&response.text).map_err(|e| GeneratorError::Enhancement {
                message: format!("Failed to parse LLM response: {e}"),
                model: Some(self.model_name.clone()),
            })?;

        let improvements = json["improvements"].as_array();
        let enhanced_front = json["enhanced_front"].as_str().unwrap_or_default();

        // Return original card if no improvements suggested
        if enhanced_front.is_empty() || improvements.is_none_or(|arr| arr.is_empty()) {
            return Ok(card.clone());
        }

        let confidence = json["confidence"]
            .as_f64()
            .unwrap_or(card.confidence as f64) as f32;
        Ok(GeneratedCard {
            confidence,
            apf_html: enhanced_front.to_string(),
            ..card.clone()
        })
    }

    #[instrument(skip_all)]
    async fn suggest_split(
        &self,
        content: &str,
        deps: &GenerationDeps,
    ) -> Result<SplitDecision, GeneratorError> {
        let prompt = format!(
            "Analyze whether this content should be split into multiple cards. \
             Note: '{}'. Content: {}",
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

        let decision: SplitDecision =
            serde_json::from_str(&response.text).map_err(|e| GeneratorError::Enhancement {
                message: format!("Failed to parse split decision: {e}"),
                model: Some(self.model_name.clone()),
            })?;

        Ok(decision)
    }
}

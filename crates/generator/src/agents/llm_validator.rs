use std::sync::Arc;

use llm::LlmProvider;

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

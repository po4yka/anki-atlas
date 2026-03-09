use std::sync::Arc;

use llm::LlmProvider;

/// LLM-backed generator agent.
pub struct LlmGeneratorAgent {
    provider: Arc<dyn LlmProvider>,
    model_name: String,
    temperature: f32,
}

impl LlmGeneratorAgent {
    pub fn new(provider: Arc<dyn LlmProvider>, model_name: String, temperature: f32) -> Self {
        Self {
            provider,
            model_name,
            temperature,
        }
    }
}

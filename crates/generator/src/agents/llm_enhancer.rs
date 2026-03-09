use std::sync::Arc;

use llm::LlmProvider;

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

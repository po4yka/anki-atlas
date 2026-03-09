pub mod llm_enhancer;
pub mod llm_generator;
pub mod llm_validator;

use async_trait::async_trait;

use crate::error::GeneratorError;
use crate::models::*;

#[cfg(test)]
mod tests;

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

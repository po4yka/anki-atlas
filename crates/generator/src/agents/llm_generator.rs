use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tracing::instrument;

use llm::LlmProvider;

use crate::agents::{GeneratorAgent, LlmAgentBase};
use crate::error::GeneratorError;
use crate::models::{GeneratedCard, GenerationDeps, GenerationResult};

/// LLM-backed generator agent.
pub struct LlmGeneratorAgent {
    base: LlmAgentBase,
}

#[derive(Deserialize)]
struct GeneratedCardPayload {
    card_index: u32,
    slug: String,
    lang: String,
    front: String,
    back: String,
    confidence: f32,
}

#[derive(Deserialize)]
struct GeneratedCardsPayload {
    cards: Vec<GeneratedCardPayload>,
}

impl LlmGeneratorAgent {
    pub fn new(provider: Arc<dyn LlmProvider>, model_name: String, temperature: f32) -> Self {
        Self {
            base: LlmAgentBase::new(provider, model_name, temperature),
        }
    }

    fn content_hash(front: &str, back: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(front.as_bytes());
        hasher.update(back.as_bytes());
        let result = hasher.finalize();
        result[..8]
            .iter()
            .fold(String::with_capacity(16), |mut s, b| {
                use std::fmt::Write;
                let _ = write!(s, "{b:02x}");
                s
            })
    }
}

#[async_trait]
impl GeneratorAgent for LlmGeneratorAgent {
    #[instrument(skip_all)]
    async fn generate(
        &self,
        deps: &GenerationDeps,
        qa_pairs: &[(String, String)],
    ) -> Result<GenerationResult, GeneratorError> {
        let start = Instant::now();

        let prompt = format!(
            "Generate Anki flashcards for note '{}' on topic '{}'. \
             Languages: {:?}. Q/A pairs: {:?}",
            deps.note_title, deps.topic, deps.language_tags, qa_pairs
        );

        let text = self.base.call_llm(&prompt).await?;

        let response: GeneratedCardsPayload =
            serde_json::from_str(&text).map_err(|e| GeneratorError::Generation {
                message: format!("Failed to parse LLM response: {e}"),
                model: Some(self.base.model_name.clone()),
            })?;

        let mut cards = Vec::with_capacity(response.cards.len());
        for raw in response.cards {
            let card = GeneratedCard {
                card_index: raw.card_index,
                slug: raw.slug,
                lang: raw.lang,
                apf_html: format!("<p>{}</p><hr><p>{}</p>", raw.front, raw.back),
                confidence: raw.confidence,
                content_hash: Self::content_hash(&raw.front, &raw.back),
            };
            cards.push(card);
        }

        let total_cards = cards.len();
        Ok(GenerationResult {
            cards,
            total_cards,
            model_used: self.base.model_name.clone(),
            generation_time_secs: start.elapsed().as_secs_f64(),
            warnings: vec![],
        })
    }
}

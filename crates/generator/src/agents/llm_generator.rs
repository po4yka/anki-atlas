use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tracing::instrument;

use llm::LlmProvider;

use taxonomy::SkillRelevance;

use crate::agents::{GeneratorAgent, LlmAgentBase};
use crate::error::GeneratorError;
use crate::models::{CardType, GeneratedCard, GenerationDeps, GenerationResult};

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
    /// Card type: "basic", "cloze", or "mcq". Defaults to basic if missing.
    #[serde(default)]
    card_type: CardType,
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

        // Skip generation for dead-skill topics
        if deps.skill_bias == Some(SkillRelevance::Dead) {
            return Ok(GenerationResult {
                cards: vec![],
                total_cards: 0,
                model_used: self.base.model_name.clone(),
                generation_time_secs: start.elapsed().as_secs_f64(),
                warnings: vec!["Skipped: dead skill topic".to_string()],
            });
        }

        let prompt = format!(
            "Generate Anki flashcards for note '{}' on topic '{}'. \
             Languages: {:?}. Q/A pairs: {:?}. \
             IMPORTANT: Generate cards that test understanding, reasoning, and application -- \
             not syntax recall or boilerplate memorization. Prefer questions about: \
             system design tradeoffs, debugging strategies, when/why to use patterns, \
             shipping and automation decisions. Each card should require thinking, not lookup. \
             \
             CARD TYPES: Generate a MIX of card types for the same source material: \
             - \"basic\": standard question/answer (front asks, back answers) \
             - \"cloze\": cloze deletion using {{{{c1::answer}}}} syntax in the front field; \
               back field should be empty or contain a hint \
             - \"mcq\": multiple choice with options labeled A/B/C/D in the front; \
               back contains the correct answer letter and explanation \
             Include the card_type field in each card object. \
             Aim for variety: at least one cloze and one basic per topic when appropriate.",
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
            let apf_html = match raw.card_type {
                CardType::Cloze => {
                    // Cloze: front contains {{c1::...}} patterns, back is hint/extra
                    if raw.back.is_empty() {
                        format!("<p>{}</p>", raw.front)
                    } else {
                        format!("<p>{}</p><hr><p>{}</p>", raw.front, raw.back)
                    }
                }
                CardType::Mcq => {
                    // MCQ: front has question + options, back has answer + explanation
                    format!(
                        "<div class=\"mcq\"><p>{}</p></div><hr><p>{}</p>",
                        raw.front, raw.back
                    )
                }
                CardType::Basic => {
                    // Standard front/back
                    format!("<p>{}</p><hr><p>{}</p>", raw.front, raw.back)
                }
            };

            let card = GeneratedCard {
                card_index: raw.card_index,
                slug: raw.slug,
                lang: raw.lang,
                apf_html,
                confidence: raw.confidence,
                content_hash: Self::content_hash(&raw.front, &raw.back),
                card_type: raw.card_type,
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

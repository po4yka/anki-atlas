use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use tracing::instrument;

use llm::{GenerateOptions, LlmProvider};

use crate::agents::GeneratorAgent;
use crate::error::GeneratorError;
use crate::models::{GeneratedCard, GenerationDeps, GenerationResult};

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
            serde_json::from_str(&response.text).map_err(|e| GeneratorError::Generation {
                message: format!("Failed to parse LLM response: {e}"),
                model: Some(self.model_name.clone()),
            })?;

        let raw_cards = json["cards"]
            .as_array()
            .ok_or_else(|| GeneratorError::Generation {
                message: "Response missing 'cards' array".into(),
                model: Some(self.model_name.clone()),
            })?;

        let mut cards = Vec::with_capacity(raw_cards.len());
        for raw in raw_cards {
            let front = raw["front"].as_str().unwrap_or_default();
            let back = raw["back"].as_str().unwrap_or_default();
            let card = GeneratedCard {
                card_index: raw["card_index"].as_u64().unwrap_or(0) as u32,
                slug: raw["slug"].as_str().unwrap_or_default().to_string(),
                lang: raw["lang"].as_str().unwrap_or_default().to_string(),
                apf_html: format!("<p>{front}</p><hr><p>{back}</p>"),
                confidence: raw["confidence"].as_f64().unwrap_or(0.0) as f32,
                content_hash: Self::content_hash(front, back),
            };
            cards.push(card);
        }

        let total_cards = cards.len();
        Ok(GenerationResult {
            cards,
            total_cards,
            model_used: self.model_name.clone(),
            generation_time_secs: start.elapsed().as_secs_f64(),
            warnings: vec![],
        })
    }
}

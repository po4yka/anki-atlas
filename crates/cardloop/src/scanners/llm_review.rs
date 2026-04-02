use card::registry::CardEntry;
use chrono::Utc;
use llm::provider::{GenerateOptions, LlmProvider};

use crate::error::CardloopError;
use crate::models::{IssueKind, ItemStatus, LoopKind, Tier, WorkItem};
use crate::scanners::AsyncScanner;

const BATCH_SIZE: usize = 10;

const SYSTEM_PROMPT: &str = "You are a flashcard quality reviewer. Score each card on 4 dimensions (0.0-1.0): \
- atomicity: Tests exactly one fact? \
- clarity: Unambiguous prompt and answer? \
- cue_discrimination: Does the prompt uniquely identify the answer? \
- pedagogical_value: Tests understanding, not trivia? \
Respond with JSON: { \"reviews\": [{ \"slug\": \"...\", \"atomicity\": 0.0, \"clarity\": 0.0, \
\"cue_discrimination\": 0.0, \"pedagogical_value\": 0.0, \"suggestion\": \"...\" }] }";

/// Scores cards using an LLM provider across 4 pedagogical dimensions.
pub struct LlmReviewScanner {
    cards: Vec<CardEntry>,
    provider: Box<dyn LlmProvider>,
    model_name: String,
}

impl LlmReviewScanner {
    pub fn new(cards: Vec<CardEntry>, provider: Box<dyn LlmProvider>, model_name: String) -> Self {
        Self {
            cards,
            provider,
            model_name,
        }
    }

    fn item_id(slug: &str, dimension: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"llm_review:");
        hasher.update(slug.as_bytes());
        hasher.update(b":");
        hasher.update(dimension.as_bytes());
        let hash = hasher.finalize();
        hash.iter().take(8).map(|b| format!("{b:02x}")).collect()
    }

    fn build_prompt(batch: &[&CardEntry]) -> String {
        let cards_json: Vec<serde_json::Value> = batch
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "slug": entry.slug,
                    "front": entry.front,
                    "back": entry.back,
                })
            })
            .collect();
        serde_json::json!({ "cards": cards_json }).to_string()
    }
}

#[derive(Debug, serde::Deserialize)]
struct ReviewResponse {
    reviews: Vec<CardReview>,
}

#[derive(Debug, serde::Deserialize)]
struct CardReview {
    slug: String,
    atomicity: f64,
    clarity: f64,
    cue_discrimination: f64,
    pedagogical_value: f64,
    suggestion: Option<String>,
}

#[async_trait::async_trait]
impl AsyncScanner for LlmReviewScanner {
    async fn scan(&self, scan_number: u32) -> Result<Vec<WorkItem>, CardloopError> {
        let opts = GenerateOptions {
            system: SYSTEM_PROMPT.to_string(),
            json_mode: true,
            temperature: 0.2,
            json_schema: None,
        };

        let mut items = Vec::new();
        let now = Utc::now();

        for batch in self.cards.chunks(BATCH_SIZE) {
            let batch_refs: Vec<&CardEntry> = batch.iter().collect();
            let prompt = Self::build_prompt(&batch_refs);

            let response = self
                .provider
                .generate(&self.model_name, &prompt, &opts)
                .await
                .map_err(|e| CardloopError::Validation(format!("LLM error: {e}")))?;

            let parsed: ReviewResponse = serde_json::from_str(&response.text).map_err(|e| {
                CardloopError::Json(
                    serde_json::from_str::<serde_json::Value>(&format!("\"parse error: {e}\""))
                        .unwrap_err(),
                )
            })?;

            for review in &parsed.reviews {
                let dimensions = [
                    ("atomicity", review.atomicity),
                    ("clarity", review.clarity),
                    ("cue_discrimination", review.cue_discrimination),
                    ("pedagogical_value", review.pedagogical_value),
                ];

                let source_path = batch
                    .iter()
                    .find(|e| e.slug == review.slug)
                    .map(|e| e.source_path.clone())
                    .unwrap_or_else(|| review.slug.clone());

                for (dim_name, dim_score) in &dimensions {
                    if *dim_score >= 0.6 {
                        continue;
                    }

                    let (tier, confidence) = if *dim_score < 0.4 {
                        (Tier::Rework, 0.7)
                    } else {
                        (Tier::QuickFix, 0.6)
                    };

                    let id = Self::item_id(&review.slug, dim_name);
                    items.push(WorkItem {
                        id,
                        loop_kind: LoopKind::Audit,
                        issue_kind: IssueKind::LowQuality {
                            dimension: dim_name.to_string(),
                            score: *dim_score,
                        },
                        tier,
                        status: ItemStatus::Open,
                        slug: Some(review.slug.clone()),
                        source_path: source_path.clone(),
                        summary: format!("Low {dim_name} ({dim_score:.2}) for {}", review.slug),
                        detail: review.suggestion.clone(),
                        first_seen: now,
                        resolved_at: None,
                        attestation: None,
                        scan_number,
                        cluster_id: None,
                        confidence: Some(confidence),
                    });
                }
            }
        }

        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use llm::error::LlmError;
    use llm::provider::GenerateOptions;
    use llm::response::LlmResponse;

    struct MockProvider {
        response_text: String,
    }

    #[async_trait::async_trait]
    impl LlmProvider for MockProvider {
        async fn generate(
            &self,
            _model: &str,
            _prompt: &str,
            _opts: &GenerateOptions,
        ) -> Result<LlmResponse, LlmError> {
            Ok(LlmResponse {
                text: self.response_text.clone(),
                model: "mock".into(),
                prompt_tokens: None,
                completion_tokens: None,
                finish_reason: None,
                raw: Default::default(),
            })
        }

        async fn check_connection(&self) -> bool {
            true
        }

        async fn list_models(&self) -> Result<Vec<String>, LlmError> {
            Ok(vec!["mock".into()])
        }
    }

    fn make_entry(slug: &str) -> CardEntry {
        let now = Utc::now();
        CardEntry {
            slug: slug.into(),
            note_id: "note-1".into(),
            source_path: "notes/test.md".into(),
            front: "What is X?".into(),
            back: "X is Y.".into(),
            content_hash: "abc".into(),
            metadata_hash: "def".into(),
            language: "en".into(),
            tags: vec!["topic::rust".into()],
            anki_note_id: None,
            created_at: Some(now),
            updated_at: Some(now),
            synced_at: None,
        }
    }

    #[tokio::test]
    async fn emits_work_items_for_low_scoring_dimensions() {
        let response = serde_json::json!({
            "reviews": [{
                "slug": "test-card",
                "atomicity": 0.3,
                "clarity": 0.7,
                "cue_discrimination": 0.5,
                "pedagogical_value": 0.8,
                "suggestion": "Split this card into two."
            }]
        })
        .to_string();

        let provider = MockProvider {
            response_text: response,
        };
        let scanner = LlmReviewScanner::new(
            vec![make_entry("test-card")],
            Box::new(provider),
            "mock-model".into(),
        );

        let items = scanner.scan(1).await.unwrap();

        // atomicity=0.3 < 0.4 → Rework; cue_discrimination=0.5 < 0.6 → QuickFix
        // clarity=0.7 and pedagogical_value=0.8 → no items
        assert_eq!(items.len(), 2);

        let atomicity_item = items.iter().find(|i| {
            matches!(&i.issue_kind, IssueKind::LowQuality { dimension, .. } if dimension == "atomicity")
        });
        assert!(atomicity_item.is_some());
        assert_eq!(atomicity_item.unwrap().tier, Tier::Rework);
        assert_eq!(
            atomicity_item.unwrap().detail.as_deref(),
            Some("Split this card into two.")
        );

        let cue_item = items.iter().find(|i| {
            matches!(&i.issue_kind, IssueKind::LowQuality { dimension, .. } if dimension == "cue_discrimination")
        });
        assert!(cue_item.is_some());
        assert_eq!(cue_item.unwrap().tier, Tier::QuickFix);
    }

    #[tokio::test]
    async fn no_items_when_all_scores_high() {
        let response = serde_json::json!({
            "reviews": [{
                "slug": "good-card",
                "atomicity": 0.9,
                "clarity": 0.85,
                "cue_discrimination": 0.75,
                "pedagogical_value": 0.95,
                "suggestion": null
            }]
        })
        .to_string();

        let provider = MockProvider {
            response_text: response,
        };
        let scanner = LlmReviewScanner::new(
            vec![make_entry("good-card")],
            Box::new(provider),
            "mock-model".into(),
        );

        let items = scanner.scan(1).await.unwrap();
        assert!(items.is_empty());
    }

    #[tokio::test]
    async fn item_ids_are_deterministic() {
        let id1 = LlmReviewScanner::item_id("my-card", "clarity");
        let id2 = LlmReviewScanner::item_id("my-card", "clarity");
        assert_eq!(id1, id2);

        let id3 = LlmReviewScanner::item_id("my-card", "atomicity");
        assert_ne!(id1, id3);
    }
}

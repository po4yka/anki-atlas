use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
#[allow(unused_imports)]
use mockall::predicate::*;

use super::*;
use crate::error::GeneratorError;

// ---------------------------------------------------------------------------
// Local mock of LlmProvider (not available from llm crate outside its tests)
// ---------------------------------------------------------------------------

mockall::mock! {
    pub TestLlmProvider {}

    #[async_trait]
    impl llm::LlmProvider for TestLlmProvider {
        async fn generate(
            &self,
            model: &str,
            prompt: &str,
            opts: &llm::GenerateOptions,
        ) -> Result<llm::LlmResponse, llm::LlmError>;

        async fn generate_json(
            &self,
            model: &str,
            prompt: &str,
            opts: &llm::GenerateOptions,
        ) -> Result<HashMap<String, serde_json::Value>, llm::LlmError>;

        async fn check_connection(&self) -> bool;

        async fn list_models(&self) -> Result<Vec<String>, llm::LlmError>;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_deps() -> GenerationDeps {
    GenerationDeps {
        note_title: "Test Note".into(),
        topic: "Rust ownership".into(),
        language_tags: vec!["en".into()],
        source_file: "test.md".into(),
    }
}

fn test_card() -> GeneratedCard {
    GeneratedCard {
        card_index: 1,
        slug: "test-card".into(),
        lang: "en".into(),
        apf_html: "<p>Hello</p>".into(),
        confidence: 0.9,
        content_hash: "abc123".into(),
    }
}

fn test_generation_result() -> GenerationResult {
    GenerationResult {
        cards: vec![test_card()],
        total_cards: 1,
        model_used: "test-model".into(),
        generation_time_secs: 1.5,
        warnings: vec![],
    }
}

fn make_llm_response(json_text: &str) -> llm::LlmResponse {
    llm::LlmResponse {
        text: json_text.to_string(),
        model: "gpt-4".into(),
        prompt_tokens: Some(100),
        completion_tokens: Some(50),
        finish_reason: Some("stop".into()),
        raw: HashMap::new(),
    }
}

// ---------------------------------------------------------------------------
// MockGeneratorAgent tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mock_generator_agent_returns_cards() {
    let mut mock = MockGeneratorAgent::new();
    mock.expect_generate()
        .with(always(), always())
        .returning(|_, _| Ok(test_generation_result()));

    let deps = test_deps();
    let pairs = vec![("Q1".to_string(), "A1".to_string())];
    let result = mock.generate(&deps, &pairs).await.unwrap();

    assert_eq!(result.total_cards, 1);
    assert_eq!(result.cards[0].slug, "test-card");
}

#[tokio::test]
async fn mock_generator_agent_returns_error() {
    let mut mock = MockGeneratorAgent::new();
    mock.expect_generate().returning(|_, _| {
        Err(GeneratorError::Generation {
            message: "LLM failed".into(),
            model: Some("gpt-4".into()),
        })
    });

    let deps = test_deps();
    let pairs = vec![("Q".into(), "A".into())];
    let err = mock.generate(&deps, &pairs).await.unwrap_err();
    assert!(matches!(err, GeneratorError::Generation { .. }));
}

#[tokio::test]
async fn mock_generator_agent_empty_qa_pairs() {
    let mut mock = MockGeneratorAgent::new();
    mock.expect_generate().returning(|_, _| {
        Ok(GenerationResult {
            cards: vec![],
            total_cards: 0,
            model_used: "test".into(),
            generation_time_secs: 0.1,
            warnings: vec!["No Q/A pairs provided".into()],
        })
    });

    let deps = test_deps();
    let result = mock.generate(&deps, &[]).await.unwrap();
    assert_eq!(result.total_cards, 0);
    assert!(result.cards.is_empty());
    assert!(!result.warnings.is_empty());
}

// ---------------------------------------------------------------------------
// MockEnhancerAgent tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mock_enhancer_agent_enhances_card() {
    let mut mock = MockEnhancerAgent::new();
    mock.expect_enhance().returning(|card, _| {
        Ok(GeneratedCard {
            confidence: 0.95,
            ..card.clone()
        })
    });

    let deps = test_deps();
    let card = test_card();
    let enhanced = mock.enhance(&card, &deps).await.unwrap();
    assert_eq!(enhanced.confidence, 0.95);
    assert_eq!(enhanced.slug, card.slug);
}

#[tokio::test]
async fn mock_enhancer_agent_returns_original_when_no_improvements() {
    let mut mock = MockEnhancerAgent::new();
    mock.expect_enhance()
        .returning(|card, _| Ok(card.clone()));

    let deps = test_deps();
    let card = test_card();
    let enhanced = mock.enhance(&card, &deps).await.unwrap();
    assert_eq!(enhanced.apf_html, card.apf_html);
    assert_eq!(enhanced.confidence, card.confidence);
}

#[tokio::test]
async fn mock_enhancer_agent_suggests_split() {
    let mut mock = MockEnhancerAgent::new();
    mock.expect_suggest_split().returning(|_, _| {
        Ok(SplitDecision {
            should_split: true,
            card_count: 2,
            plans: vec![
                SplitPlan {
                    card_number: 1,
                    concept: "Ownership basics".into(),
                    question: "What is ownership?".into(),
                    answer_summary: "Each value has one owner".into(),
                },
                SplitPlan {
                    card_number: 2,
                    concept: "Borrowing".into(),
                    question: "What is borrowing?".into(),
                    answer_summary: "References allow access without ownership".into(),
                },
            ],
            reasoning: "Content covers two distinct concepts".into(),
        })
    });

    let deps = test_deps();
    let result = mock.suggest_split("long content", &deps).await.unwrap();
    assert!(result.should_split);
    assert_eq!(result.card_count, 2);
    assert_eq!(result.plans.len(), 2);
}

#[tokio::test]
async fn mock_enhancer_agent_suggests_no_split() {
    let mut mock = MockEnhancerAgent::new();
    mock.expect_suggest_split().returning(|_, _| {
        Ok(SplitDecision {
            should_split: false,
            card_count: 1,
            plans: vec![],
            reasoning: "Content is focused".into(),
        })
    });

    let deps = test_deps();
    let result = mock.suggest_split("short", &deps).await.unwrap();
    assert!(!result.should_split);
    assert_eq!(result.card_count, 1);
    assert!(result.plans.is_empty());
}

#[tokio::test]
async fn mock_enhancer_agent_enhance_error() {
    let mut mock = MockEnhancerAgent::new();
    mock.expect_enhance().returning(|_, _| {
        Err(GeneratorError::Enhancement {
            message: "Enhancement failed".into(),
            model: None,
        })
    });

    let deps = test_deps();
    let card = test_card();
    let err = mock.enhance(&card, &deps).await.unwrap_err();
    assert!(matches!(err, GeneratorError::Enhancement { .. }));
}

// ---------------------------------------------------------------------------
// MockValidatorAgent tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mock_validator_agent_valid_content() {
    let mut mock = MockValidatorAgent::new();
    mock.expect_validate()
        .returning(|_, _| Ok(ValidationResult { issues: vec![] }));

    let deps = test_deps();
    let result = mock.validate("good content", &deps).await.unwrap();
    assert!(result.is_valid());
    assert!(result.errors().is_empty());
}

#[tokio::test]
async fn mock_validator_agent_finds_issues() {
    let mut mock = MockValidatorAgent::new();
    mock.expect_validate().returning(|_, _| {
        Ok(ValidationResult {
            issues: vec![
                ValidationIssue {
                    severity: Severity::Error,
                    message: "Content too short".into(),
                    location: None,
                },
                ValidationIssue {
                    severity: Severity::Warning,
                    message: "Consider adding examples".into(),
                    location: Some("body".into()),
                },
            ],
        })
    });

    let deps = test_deps();
    let result = mock.validate("bad", &deps).await.unwrap();
    assert!(!result.is_valid());
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.warnings().len(), 1);
}

#[tokio::test]
async fn mock_validator_agent_returns_error() {
    let mut mock = MockValidatorAgent::new();
    mock.expect_validate().returning(|_, _| {
        Err(GeneratorError::Validation {
            message: "Validation service unavailable".into(),
        })
    });

    let deps = test_deps();
    let err = mock.validate("content", &deps).await.unwrap_err();
    assert!(matches!(err, GeneratorError::Validation { .. }));
}

// ---------------------------------------------------------------------------
// Send + Sync assertions for trait objects
// ---------------------------------------------------------------------------

fn _assert_generator_send_sync(_: impl GeneratorAgent) {}
fn _assert_enhancer_send_sync(_: impl EnhancerAgent) {}
fn _assert_validator_send_sync(_: impl ValidatorAgent) {}

#[test]
fn trait_objects_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Box<dyn GeneratorAgent>>();
    assert_send_sync::<Box<dyn EnhancerAgent>>();
    assert_send_sync::<Box<dyn ValidatorAgent>>();
    assert_send_sync::<Arc<dyn GeneratorAgent>>();
    assert_send_sync::<Arc<dyn EnhancerAgent>>();
    assert_send_sync::<Arc<dyn ValidatorAgent>>();
}

// ---------------------------------------------------------------------------
// LLM-backed agent struct tests (RED - these modules don't exist yet)
// ---------------------------------------------------------------------------

use super::llm_enhancer::LlmEnhancerAgent;
use super::llm_generator::LlmGeneratorAgent;
use super::llm_validator::{LlmPostValidatorAgent, LlmPreValidatorAgent};

#[test]
fn llm_generator_agent_new() {
    let provider: Arc<dyn llm::LlmProvider> = Arc::new(MockTestLlmProvider::new());
    let agent = LlmGeneratorAgent::new(provider, "gpt-4".to_string(), 0.3);
    let _ = &agent;
}

#[test]
fn llm_enhancer_agent_new() {
    let provider: Arc<dyn llm::LlmProvider> = Arc::new(MockTestLlmProvider::new());
    let agent = LlmEnhancerAgent::new(provider, "gpt-4".to_string(), 0.3);
    let _ = &agent;
}

#[test]
fn llm_pre_validator_agent_new() {
    let provider: Arc<dyn llm::LlmProvider> = Arc::new(MockTestLlmProvider::new());
    let agent = LlmPreValidatorAgent::new(provider, "gpt-4".to_string(), 0.0);
    let _ = &agent;
}

#[test]
fn llm_post_validator_agent_new() {
    let provider: Arc<dyn llm::LlmProvider> = Arc::new(MockTestLlmProvider::new());
    let agent = LlmPostValidatorAgent::new(provider, "gpt-4".to_string(), 0.0);
    let _ = &agent;
}

#[test]
fn llm_agents_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<LlmGeneratorAgent>();
    assert_send_sync::<LlmEnhancerAgent>();
    assert_send_sync::<LlmPreValidatorAgent>();
    assert_send_sync::<LlmPostValidatorAgent>();
}

#[tokio::test]
async fn llm_generator_agent_generates_cards_via_mock_provider() {
    let mut mock_provider = MockTestLlmProvider::new();
    mock_provider.expect_generate().returning(|_, _, _| {
        Ok(make_llm_response(
            &serde_json::json!({
                "cards": [{
                    "card_index": 1,
                    "slug": "ownership-basics",
                    "lang": "en",
                    "front": "What is ownership in Rust?",
                    "back": "Each value has exactly one owner.",
                    "confidence": 0.9
                }]
            })
            .to_string(),
        ))
    });

    let provider: Arc<dyn llm::LlmProvider> = Arc::new(mock_provider);
    let agent = LlmGeneratorAgent::new(provider, "gpt-4".into(), 0.3);
    let deps = test_deps();
    let qa_pairs = vec![(
        "What is ownership?".to_string(),
        "Each value has one owner".to_string(),
    )];
    let result = agent.generate(&deps, &qa_pairs).await.unwrap();

    assert!(!result.cards.is_empty());
    assert_eq!(result.model_used, "gpt-4");
    assert!(result.generation_time_secs >= 0.0);
}

#[tokio::test]
async fn llm_generator_agent_computes_content_hash() {
    let mut mock_provider = MockTestLlmProvider::new();
    mock_provider.expect_generate().returning(|_, _, _| {
        Ok(make_llm_response(
            &serde_json::json!({
                "cards": [{
                    "card_index": 1,
                    "slug": "hash-test",
                    "lang": "en",
                    "front": "Q",
                    "back": "A",
                    "confidence": 0.8
                }]
            })
            .to_string(),
        ))
    });

    let provider: Arc<dyn llm::LlmProvider> = Arc::new(mock_provider);
    let agent = LlmGeneratorAgent::new(provider, "gpt-4".into(), 0.3);
    let deps = test_deps();
    let qa_pairs = vec![("Q".into(), "A".into())];
    let result = agent.generate(&deps, &qa_pairs).await.unwrap();

    for card in &result.cards {
        // content_hash should be 16 hex chars (first 16 of SHA-256)
        assert_eq!(card.content_hash.len(), 16);
        assert!(card.content_hash.chars().all(|c| c.is_ascii_hexdigit()));
    }
}

#[tokio::test]
async fn llm_enhancer_agent_returns_original_when_no_improvements() {
    let mut mock_provider = MockTestLlmProvider::new();
    mock_provider.expect_generate().returning(|_, _, _| {
        Ok(make_llm_response(
            &serde_json::json!({
                "enhanced_front": "",
                "improvements": [],
                "confidence": 0.9
            })
            .to_string(),
        ))
    });

    let provider: Arc<dyn llm::LlmProvider> = Arc::new(mock_provider);
    let agent = LlmEnhancerAgent::new(provider, "gpt-4".into(), 0.3);
    let deps = test_deps();
    let card = test_card();
    let enhanced = agent.enhance(&card, &deps).await.unwrap();

    // Should return original card when no improvements suggested
    assert_eq!(enhanced.apf_html, card.apf_html);
}

#[tokio::test]
async fn llm_enhancer_agent_suggest_split_parses_plans() {
    let mut mock_provider = MockTestLlmProvider::new();
    mock_provider.expect_generate().returning(|_, _, _| {
        Ok(make_llm_response(
            &serde_json::json!({
                "should_split": true,
                "card_count": 2,
                "plans": [
                    {
                        "card_number": 1,
                        "concept": "Ownership",
                        "question": "What is ownership?",
                        "answer_summary": "One owner per value"
                    },
                    {
                        "card_number": 2,
                        "concept": "Borrowing",
                        "question": "What is borrowing?",
                        "answer_summary": "Temporary access via references"
                    }
                ],
                "reasoning": "Two distinct concepts"
            })
            .to_string(),
        ))
    });

    let provider: Arc<dyn llm::LlmProvider> = Arc::new(mock_provider);
    let agent = LlmEnhancerAgent::new(provider, "gpt-4".into(), 0.3);
    let deps = test_deps();
    let decision = agent
        .suggest_split("long content about ownership and borrowing", &deps)
        .await
        .unwrap();

    assert!(decision.should_split);
    assert_eq!(decision.card_count, 2);
    assert_eq!(decision.plans.len(), 2);
    assert_eq!(decision.plans[0].concept, "Ownership");
    assert_eq!(decision.plans[1].concept, "Borrowing");
}

#[tokio::test]
async fn llm_generator_agent_propagates_llm_error() {
    let mut mock_provider = MockTestLlmProvider::new();
    mock_provider.expect_generate().returning(|_, _, _| {
        Err(llm::LlmError::Http {
            status: 429,
            body: "rate limited".into(),
        })
    });

    let provider: Arc<dyn llm::LlmProvider> = Arc::new(mock_provider);
    let agent = LlmGeneratorAgent::new(provider, "gpt-4".into(), 0.3);
    let deps = test_deps();
    let err = agent
        .generate(&deps, &[("Q".into(), "A".into())])
        .await
        .unwrap_err();
    assert!(matches!(err, GeneratorError::Llm(_)));
}

// ---------------------------------------------------------------------------
// LLM agents implement the correct trait
// ---------------------------------------------------------------------------

#[test]
fn llm_generator_agent_implements_generator_trait() {
    fn accepts_generator(_: &dyn GeneratorAgent) {}
    let provider: Arc<dyn llm::LlmProvider> = Arc::new(MockTestLlmProvider::new());
    let agent = LlmGeneratorAgent::new(provider, "m".into(), 0.3);
    accepts_generator(&agent);
}

#[test]
fn llm_enhancer_agent_implements_enhancer_trait() {
    fn accepts_enhancer(_: &dyn EnhancerAgent) {}
    let provider: Arc<dyn llm::LlmProvider> = Arc::new(MockTestLlmProvider::new());
    let agent = LlmEnhancerAgent::new(provider, "m".into(), 0.3);
    accepts_enhancer(&agent);
}

#[test]
fn llm_pre_validator_implements_validator_trait() {
    fn accepts_validator(_: &dyn ValidatorAgent) {}
    let provider: Arc<dyn llm::LlmProvider> = Arc::new(MockTestLlmProvider::new());
    let agent = LlmPreValidatorAgent::new(provider, "m".into(), 0.0);
    accepts_validator(&agent);
}

#[test]
fn llm_post_validator_implements_validator_trait() {
    fn accepts_validator(_: &dyn ValidatorAgent) {}
    let provider: Arc<dyn llm::LlmProvider> = Arc::new(MockTestLlmProvider::new());
    let agent = LlmPostValidatorAgent::new(provider, "m".into(), 0.0);
    accepts_validator(&agent);
}

use std::collections::HashMap;
use std::sync::Arc;

use super::*;

// ─── GenerateOptions ──────────────────────────────────────

#[test]
fn generate_options_default_has_expected_values() {
    let opts = GenerateOptions::default();
    assert_eq!(opts.system, "");
    assert!(
        (opts.temperature - 0.7).abs() < f32::EPSILON,
        "default temperature should be 0.7"
    );
    assert!(!opts.json_mode);
    assert!(opts.json_schema.is_none());
}

#[test]
fn generate_options_clone_is_independent() {
    let opts = GenerateOptions {
        system: "test system".to_string(),
        temperature: 0.5,
        json_mode: true,
        json_schema: Some(serde_json::json!({"type": "object"})),
    };
    let cloned = opts.clone();
    assert_eq!(cloned.system, "test system");
    assert!((cloned.temperature - 0.5).abs() < f32::EPSILON);
    assert!(cloned.json_mode);
    assert!(cloned.json_schema.is_some());
}

#[test]
fn generate_options_debug_format() {
    let opts = GenerateOptions::default();
    let debug = format!("{opts:?}");
    assert!(debug.contains("GenerateOptions"));
}

// ─── MockLlmProvider ──────────────────────────────────────

#[tokio::test]
async fn mock_provider_generate_returns_configured_response() {
    let mut mock = MockLlmProvider::new();
    mock.expect_generate().returning(|_model, _prompt, _opts| {
        Box::pin(async {
            Ok(LlmResponse {
                text: "Hello from mock".to_string(),
                model: "test-model".to_string(),
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
                finish_reason: Some("stop".to_string()),
                raw: HashMap::new(),
            })
        })
    });

    let opts = GenerateOptions::default();
    let result = mock.generate("test-model", "Hello", &opts).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.text, "Hello from mock");
    assert_eq!(response.model, "test-model");
    assert_eq!(response.total_tokens(), Some(30));
}

#[tokio::test]
async fn mock_provider_generate_can_return_error() {
    let mut mock = MockLlmProvider::new();
    mock.expect_generate().returning(|_model, _prompt, _opts| {
        Box::pin(async { Err(LlmError::Connection("test error".to_string())) })
    });

    let opts = GenerateOptions::default();
    let result = mock.generate("model", "prompt", &opts).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn mock_provider_check_connection() {
    let mut mock = MockLlmProvider::new();
    mock.expect_check_connection()
        .returning(|| Box::pin(async { true }));
    assert!(mock.check_connection().await);
}

#[tokio::test]
async fn mock_provider_list_models() {
    let mut mock = MockLlmProvider::new();
    mock.expect_list_models()
        .returning(|| Box::pin(async { Ok(vec!["model-a".to_string(), "model-b".to_string()]) }));

    let models = mock.list_models().await.unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0], "model-a");
}

// ─── Send + Sync assertions ──────────────────────────────

#[test]
fn generate_options_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GenerateOptions>();
}

#[test]
fn llm_provider_trait_object_is_send_sync() {
    fn assert_send_sync<T: Send + Sync + ?Sized>() {}
    assert_send_sync::<dyn LlmProvider>();
}

#[test]
fn provider_can_be_wrapped_in_arc() {
    fn assert_arc_compatible<T: Send + Sync + ?Sized>() {}
    assert_arc_compatible::<Arc<dyn LlmProvider>>();
}

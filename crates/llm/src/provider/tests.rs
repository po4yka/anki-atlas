use std::collections::HashMap;
use std::sync::Arc;

use super::*;

// ─── GenerateOptions ──────────────────────────────────────

#[test]
fn generate_options_default_has_expected_values() {
    let opts = GenerateOptions::default();
    assert_eq!(opts.system, "");
    assert!((opts.temperature - 0.7).abs() < f32::EPSILON, "default temperature should be 0.7");
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
    mock.expect_generate()
        .returning(|_model, _prompt, _opts| {
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
    mock.expect_generate()
        .returning(|_model, _prompt, _opts| {
            Box::pin(async {
                Err(LlmError::Connection("test error".to_string()))
            })
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
    mock.expect_list_models().returning(|| {
        Box::pin(async {
            Ok(vec!["model-a".to_string(), "model-b".to_string()])
        })
    });

    let models = mock.list_models().await.unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0], "model-a");
}

// ─── generate_json default implementation ─────────────────
// We use a concrete test struct to exercise the default impl,
// since MockLlmProvider overrides all methods including defaults.

/// A test provider that only implements the required methods,
/// relying on the default `generate_json` implementation.
struct StubProvider {
    generate_response: Result<LlmResponse, LlmError>,
}

#[async_trait]
impl LlmProvider for StubProvider {
    async fn generate(
        &self,
        _model: &str,
        _prompt: &str,
        _opts: &GenerateOptions,
    ) -> Result<LlmResponse, LlmError> {
        match &self.generate_response {
            Ok(r) => Ok(r.clone()),
            Err(LlmError::Http { status, body }) => Err(LlmError::Http {
                status: *status,
                body: body.clone(),
            }),
            Err(LlmError::Connection(msg)) => Err(LlmError::Connection(msg.clone())),
            _ => Err(LlmError::Connection("unexpected".to_string())),
        }
    }

    async fn check_connection(&self) -> bool {
        true
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        Ok(vec![])
    }
}

#[tokio::test]
async fn generate_json_parses_valid_json_response() {
    let provider = StubProvider {
        generate_response: Ok(LlmResponse {
            text: r#"{"key": "value", "count": 42}"#.to_string(),
            model: "test-model".to_string(),
            prompt_tokens: Some(5),
            completion_tokens: Some(10),
            finish_reason: Some("stop".to_string()),
            raw: HashMap::new(),
        }),
    };

    let opts = GenerateOptions::default();
    let result = provider.generate_json("test-model", "give json", &opts).await;
    assert!(result.is_ok());
    let map = result.unwrap();
    assert_eq!(map.get("key").unwrap(), "value");
    assert_eq!(map.get("count").unwrap(), 42);
}

#[tokio::test]
async fn generate_json_sets_json_mode_true() {
    // We need a provider that checks json_mode was set to true
    struct JsonModeChecker;

    #[async_trait]
    impl LlmProvider for JsonModeChecker {
        async fn generate(
            &self,
            _model: &str,
            _prompt: &str,
            opts: &GenerateOptions,
        ) -> Result<LlmResponse, LlmError> {
            assert!(opts.json_mode, "generate_json should set json_mode = true");
            Ok(LlmResponse {
                text: r#"{"ok": true}"#.to_string(),
                model: "m".to_string(),
                prompt_tokens: None,
                completion_tokens: None,
                finish_reason: None,
                raw: HashMap::new(),
            })
        }

        async fn check_connection(&self) -> bool {
            true
        }

        async fn list_models(&self) -> Result<Vec<String>, LlmError> {
            Ok(vec![])
        }
    }

    let provider = JsonModeChecker;
    let opts = GenerateOptions {
        json_mode: false, // should be overridden to true
        ..GenerateOptions::default()
    };
    let result = provider.generate_json("m", "p", &opts).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn generate_json_returns_invalid_json_error_on_non_json() {
    let provider = StubProvider {
        generate_response: Ok(LlmResponse {
            text: "This is not JSON at all".to_string(),
            model: "m".to_string(),
            prompt_tokens: None,
            completion_tokens: None,
            finish_reason: None,
            raw: HashMap::new(),
        }),
    };

    let opts = GenerateOptions::default();
    let result = provider.generate_json("m", "p", &opts).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        LlmError::InvalidJson {
            message,
            response_text,
        } => {
            assert!(message.contains("invalid JSON"));
            assert_eq!(response_text, "This is not JSON at all");
        }
        other => panic!("expected InvalidJson, got: {other:?}"),
    }
}

#[tokio::test]
async fn generate_json_truncates_long_response_text_in_error() {
    let long_text = "x".repeat(1000);
    let provider = StubProvider {
        generate_response: Ok(LlmResponse {
            text: long_text,
            model: "m".to_string(),
            prompt_tokens: None,
            completion_tokens: None,
            finish_reason: None,
            raw: HashMap::new(),
        }),
    };

    let opts = GenerateOptions::default();
    let result = provider.generate_json("m", "p", &opts).await;
    match result.unwrap_err() {
        LlmError::InvalidJson { response_text, .. } => {
            assert!(
                response_text.len() <= 500,
                "response_text should be truncated to 500 chars, got {}",
                response_text.len()
            );
        }
        other => panic!("expected InvalidJson, got: {other:?}"),
    }
}

#[tokio::test]
async fn generate_json_propagates_generate_error() {
    let provider = StubProvider {
        generate_response: Err(LlmError::Http {
            status: 500,
            body: "server error".to_string(),
        }),
    };

    let opts = GenerateOptions::default();
    let result = provider.generate_json("m", "p", &opts).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        LlmError::Http { status, .. } => assert_eq!(status, 500),
        other => panic!("expected Http error, got: {other:?}"),
    }
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

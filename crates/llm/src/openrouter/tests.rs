use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::error::LlmError;
use crate::openrouter::{OpenRouterConfig, OpenRouterProvider};
use crate::provider::{GenerateOptions, LlmProvider};

// -- Helpers --

const TEST_API_KEY: &str = "test-key-123";

fn config_with_url(base_url: &str) -> OpenRouterConfig {
    OpenRouterConfig {
        api_key: TEST_API_KEY.to_string(),
        base_url: base_url.to_string(),
        timeout_secs: 5,
        max_tokens: 1024,
        max_retries: 3,
        site_url: None,
        site_name: None,
    }
}

fn success_response() -> serde_json::Value {
    json!({
        "id": "gen-abc123",
        "model": "openai/gpt-4",
        "choices": [{
            "message": { "role": "assistant", "content": "Hello world" },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5
        }
    })
}

// -- Construction tests --

#[test]
fn new_returns_error_when_api_key_is_empty() {
    let config = OpenRouterConfig {
        api_key: String::new(),
        ..OpenRouterConfig::default()
    };
    match OpenRouterProvider::new(config) {
        Err(LlmError::Provider { message, .. }) => {
            assert!(
                message.to_lowercase().contains("api key"),
                "error message should mention API key, got: {message}"
            );
        }
        Err(other) => panic!("expected Provider error, got: {other:?}"),
        Ok(_) => panic!("expected error for empty API key"),
    }
}

#[test]
fn new_succeeds_with_valid_api_key() {
    let config = OpenRouterConfig {
        api_key: TEST_API_KEY.to_string(),
        ..OpenRouterConfig::default()
    };
    let result = OpenRouterProvider::new(config);
    assert!(result.is_ok());
}

#[test]
fn default_config_has_expected_values() {
    let config = OpenRouterConfig::default();
    assert_eq!(config.base_url, "https://openrouter.ai/api/v1");
    assert_eq!(config.timeout_secs, 180);
    assert_eq!(config.max_tokens, 2048);
    assert_eq!(config.max_retries, 3);
    assert!(config.site_url.is_none());
    assert!(config.site_name.is_none());
}

// -- Generate: message building (wiremock) --

#[tokio::test]
async fn generate_sends_user_message_only_when_no_system() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("Authorization", &format!("Bearer {TEST_API_KEY}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new(config_with_url(&server.uri())).unwrap();
    let opts = GenerateOptions {
        system: String::new(),
        ..GenerateOptions::default()
    };

    let result = provider.generate("openai/gpt-4", "Say hello", &opts).await;
    assert!(result.is_ok(), "expected Ok, got: {result:?}");
}

#[tokio::test]
async fn generate_sends_system_and_user_messages_when_system_provided() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new(config_with_url(&server.uri())).unwrap();
    let opts = GenerateOptions {
        system: "You are helpful.".to_string(),
        ..GenerateOptions::default()
    };

    let result = provider.generate("openai/gpt-4", "Say hello", &opts).await;
    assert!(result.is_ok(), "expected Ok, got: {result:?}");
}

// -- Response parsing --

#[tokio::test]
async fn generate_extracts_text_tokens_and_finish_reason() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new(config_with_url(&server.uri())).unwrap();
    let result = provider
        .generate("openai/gpt-4", "Say hello", &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.text, "Hello world");
    assert_eq!(result.model, "openai/gpt-4");
    assert_eq!(result.prompt_tokens, Some(10));
    assert_eq!(result.completion_tokens, Some(5));
    assert_eq!(result.finish_reason.as_deref(), Some("stop"));
}

#[tokio::test]
async fn generate_uses_fallback_model_when_not_in_response() {
    let server = MockServer::start().await;

    let response = json!({
        "choices": [{
            "message": { "role": "assistant", "content": "Hi" },
            "finish_reason": "stop"
        }],
        "usage": {}
    });

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new(config_with_url(&server.uri())).unwrap();
    let result = provider
        .generate("my-model", "Hi", &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.model, "my-model");
}

#[tokio::test]
async fn generate_returns_error_on_empty_choices() {
    let server = MockServer::start().await;

    let response = json!({
        "choices": [],
        "usage": {}
    });

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new(config_with_url(&server.uri())).unwrap();
    let result = provider
        .generate("openai/gpt-4", "Hi", &GenerateOptions::default())
        .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        LlmError::Provider { message, .. } => {
            assert!(
                message.to_lowercase().contains("choices"),
                "error should mention empty choices, got: {message}"
            );
        }
        other => panic!("expected Provider error, got: {other:?}"),
    }
}

#[tokio::test]
async fn generate_handles_missing_usage_fields() {
    let server = MockServer::start().await;

    let response = json!({
        "model": "openai/gpt-4",
        "choices": [{
            "message": { "role": "assistant", "content": "Ok" },
            "finish_reason": "stop"
        }]
    });

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new(config_with_url(&server.uri())).unwrap();
    let result = provider
        .generate("openai/gpt-4", "Hi", &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.prompt_tokens, None);
    assert_eq!(result.completion_tokens, None);
}

#[tokio::test]
async fn generate_rejects_missing_message_content() {
    let server = MockServer::start().await;

    let response = json!({
        "model": "openai/gpt-4",
        "choices": [{
            "message": { "role": "assistant" },
            "finish_reason": "stop"
        }]
    });

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(response))
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new(config_with_url(&server.uri())).unwrap();
    let result = provider
        .generate("openai/gpt-4", "Hi", &GenerateOptions::default())
        .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        LlmError::Provider { message, .. } => {
            assert!(message.contains("assistant message content"));
        }
        other => panic!("expected Provider error, got: {other:?}"),
    }
}

// -- Retry logic --

#[tokio::test]
async fn generate_retries_on_429_then_succeeds() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .up_to_n_times(1)
        .expect(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
        .expect(1)
        .mount(&server)
        .await;

    let mut config = config_with_url(&server.uri());
    config.max_retries = 3;
    let provider = OpenRouterProvider::new(config).unwrap();

    let result = provider
        .generate("openai/gpt-4", "Hi", &GenerateOptions::default())
        .await;

    assert!(
        result.is_ok(),
        "should succeed after retry, got: {result:?}"
    );
}

#[tokio::test]
async fn generate_retries_on_502_503_504() {
    for status in [502, 503, 504] {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(status).set_body_string("server error"))
            .up_to_n_times(1)
            .mount(&server)
            .await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
            .mount(&server)
            .await;

        let mut config = config_with_url(&server.uri());
        config.max_retries = 3;
        let provider = OpenRouterProvider::new(config).unwrap();

        let result = provider
            .generate("openai/gpt-4", "Hi", &GenerateOptions::default())
            .await;

        assert!(
            result.is_ok(),
            "should retry on {status} and succeed, got: {result:?}"
        );
    }
}

#[tokio::test]
async fn generate_exhausted_after_max_retries() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .expect(3)
        .mount(&server)
        .await;

    let mut config = config_with_url(&server.uri());
    config.max_retries = 3;
    let provider = OpenRouterProvider::new(config).unwrap();

    let result = provider
        .generate("openai/gpt-4", "Hi", &GenerateOptions::default())
        .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        LlmError::Exhausted { attempts, .. } => {
            assert_eq!(attempts, 3);
        }
        other => panic!("expected Exhausted error, got: {other:?}"),
    }
}

// -- Non-retryable errors --

#[tokio::test]
async fn generate_fails_immediately_on_400() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new(config_with_url(&server.uri())).unwrap();
    let result = provider
        .generate("openai/gpt-4", "Hi", &GenerateOptions::default())
        .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        LlmError::Http { status, body } => {
            assert_eq!(status, 400);
            assert!(body.contains("bad request"));
        }
        other => panic!("expected Http error, got: {other:?}"),
    }
}

#[tokio::test]
async fn generate_fails_immediately_on_401() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new(config_with_url(&server.uri())).unwrap();
    let result = provider
        .generate("openai/gpt-4", "Hi", &GenerateOptions::default())
        .await;

    assert!(result.is_err());
    match result.unwrap_err() {
        LlmError::Http { status, .. } => {
            assert_eq!(status, 401);
        }
        other => panic!("expected Http error, got: {other:?}"),
    }
}

// -- check_connection --

#[tokio::test]
async fn check_connection_returns_true_on_200() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": []})))
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new(config_with_url(&server.uri())).unwrap();
    assert!(provider.check_connection().await);
}

#[tokio::test]
async fn check_connection_returns_false_on_error() {
    let config = OpenRouterConfig {
        api_key: TEST_API_KEY.to_string(),
        base_url: "http://127.0.0.1:1".to_string(),
        timeout_secs: 1,
        ..OpenRouterConfig::default()
    };
    let provider = OpenRouterProvider::new(config).unwrap();
    assert!(!provider.check_connection().await);
}

// -- list_models --

#[tokio::test]
async fn list_models_parses_model_ids() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                {"id": "openai/gpt-4"},
                {"id": "anthropic/claude-3-opus"},
                {"id": "meta-llama/llama-3-70b"}
            ]
        })))
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new(config_with_url(&server.uri())).unwrap();
    let models = provider.list_models().await.unwrap();

    assert_eq!(models.len(), 3);
    assert!(models.contains(&"openai/gpt-4".to_string()));
    assert!(models.contains(&"anthropic/claude-3-opus".to_string()));
    assert!(models.contains(&"meta-llama/llama-3-70b".to_string()));
}

#[tokio::test]
async fn list_models_returns_error_on_connection_failure() {
    let config = OpenRouterConfig {
        api_key: TEST_API_KEY.to_string(),
        base_url: "http://127.0.0.1:1".to_string(),
        timeout_secs: 1,
        ..OpenRouterConfig::default()
    };
    let provider = OpenRouterProvider::new(config).unwrap();
    let result = provider.list_models().await;
    assert!(result.is_err());
}

#[tokio::test]
async fn list_models_rejects_missing_data_array() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "unexpected": []
        })))
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new(config_with_url(&server.uri())).unwrap();
    let result = provider.list_models().await;
    assert!(matches!(result, Err(LlmError::Provider { .. })));
}

// -- Headers --

#[tokio::test]
async fn generate_sends_authorization_header() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("Authorization", &format!("Bearer {TEST_API_KEY}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OpenRouterProvider::new(config_with_url(&server.uri())).unwrap();
    let result = provider
        .generate("openai/gpt-4", "Hi", &GenerateOptions::default())
        .await;

    assert!(
        result.is_ok(),
        "expected Ok with auth header, got: {result:?}"
    );
}

#[tokio::test]
async fn generate_sends_site_headers_when_configured() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("HTTP-Referer", "https://example.com"))
        .and(header("X-Title", "My App"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
        .expect(1)
        .mount(&server)
        .await;

    let config = OpenRouterConfig {
        api_key: TEST_API_KEY.to_string(),
        base_url: server.uri(),
        timeout_secs: 5,
        max_tokens: 1024,
        max_retries: 3,
        site_url: Some("https://example.com".to_string()),
        site_name: Some("My App".to_string()),
    };
    let provider = OpenRouterProvider::new(config).unwrap();
    let result = provider
        .generate("openai/gpt-4", "Hi", &GenerateOptions::default())
        .await;

    assert!(
        result.is_ok(),
        "expected Ok with site headers, got: {result:?}"
    );
}

// -- Send + Sync --

#[test]
fn openrouter_provider_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<OpenRouterProvider>();
}

#[test]
fn openrouter_provider_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<OpenRouterProvider>();
}

#[test]
fn openrouter_config_is_clone_and_debug() {
    let config = OpenRouterConfig::default();
    let _cloned = config.clone();
    let _debug = format!("{config:?}");
}

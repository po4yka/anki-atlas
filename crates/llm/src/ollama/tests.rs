use serde_json::json;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::error::LlmError;
use crate::ollama::{OllamaConfig, OllamaProvider};
use crate::provider::{GenerateOptions, LlmProvider};

// --- Helpers ---

fn success_response() -> serde_json::Value {
    json!({
        "response": "Hello world",
        "model": "llama3:latest",
        "prompt_eval_count": 10,
        "eval_count": 20,
        "done": true,
        "done_reason": "stop"
    })
}

// --- Config defaults ---

#[test]
fn config_defaults() {
    let config = OllamaConfig::default();
    assert_eq!(config.base_url, "http://localhost:11434");
    assert!(config.api_key.is_none());
    assert_eq!(config.timeout_secs, 300);
}

#[test]
fn config_clone_and_debug() {
    let config = OllamaConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.base_url, config.base_url);
    let debug = format!("{config:?}");
    assert!(debug.contains("OllamaConfig"));
}

// --- Construction ---

#[test]
fn new_creates_provider() {
    let config = OllamaConfig::default();
    let _provider = OllamaProvider::new(config);
}

#[test]
fn new_with_custom_config() {
    let config = OllamaConfig {
        base_url: "http://remote:11434".to_string(),
        api_key: Some("test-key".to_string()),
        timeout_secs: 60,
    };
    let _provider = OllamaProvider::new(config);
}

// --- Generate: payload building ---

#[tokio::test]
async fn generate_sends_correct_payload() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .and(body_partial_json(json!({
            "model": "llama3",
            "prompt": "Hello",
            "stream": false
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OllamaProvider::new(OllamaConfig {
        base_url: server.uri(),
        ..Default::default()
    });

    let opts = GenerateOptions::default();
    let result = provider.generate("llama3", "Hello", &opts).await;
    assert!(result.is_ok(), "expected Ok, got: {result:?}");
}

#[tokio::test]
async fn generate_includes_system_prompt_when_non_empty() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .and(body_partial_json(json!({
            "system": "You are helpful"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OllamaProvider::new(OllamaConfig {
        base_url: server.uri(),
        ..Default::default()
    });

    let opts = GenerateOptions {
        system: "You are helpful".to_string(),
        ..Default::default()
    };
    let result = provider.generate("llama3", "Hi", &opts).await;
    assert!(result.is_ok(), "expected Ok, got: {result:?}");
}

#[tokio::test]
async fn generate_includes_temperature_in_options() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .and(body_partial_json(json!({
            "options": { "temperature": 0.5 }
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OllamaProvider::new(OllamaConfig {
        base_url: server.uri(),
        ..Default::default()
    });

    let opts = GenerateOptions {
        temperature: 0.5,
        ..Default::default()
    };
    let result = provider.generate("llama3", "test", &opts).await;
    assert!(result.is_ok(), "expected Ok, got: {result:?}");
}

#[tokio::test]
async fn generate_sets_json_format_when_json_mode() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .and(body_partial_json(json!({
            "format": "json"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "response": "{\"key\": \"value\"}",
            "model": "llama3",
            "done": true
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OllamaProvider::new(OllamaConfig {
        base_url: server.uri(),
        ..Default::default()
    });

    let opts = GenerateOptions {
        json_mode: true,
        ..Default::default()
    };
    let result = provider.generate("llama3", "test", &opts).await;
    assert!(result.is_ok(), "expected Ok, got: {result:?}");
}

// --- Generate: response parsing ---

#[tokio::test]
async fn generate_parses_full_response() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "response": "Hello world",
            "model": "llama3:latest",
            "prompt_eval_count": 10,
            "eval_count": 20,
            "done": true,
            "done_reason": "stop"
        })))
        .mount(&server)
        .await;

    let provider = OllamaProvider::new(OllamaConfig {
        base_url: server.uri(),
        ..Default::default()
    });

    let result = provider
        .generate("llama3", "Hi", &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.text, "Hello world");
    assert_eq!(result.model, "llama3:latest");
    assert_eq!(result.prompt_tokens, Some(10));
    assert_eq!(result.completion_tokens, Some(20));
    assert_eq!(result.finish_reason, Some("stop".to_string()));
}

#[tokio::test]
async fn generate_handles_missing_optional_fields() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "response": "test",
            "model": "llama3",
            "done": true
        })))
        .mount(&server)
        .await;

    let provider = OllamaProvider::new(OllamaConfig {
        base_url: server.uri(),
        ..Default::default()
    });

    let result = provider
        .generate("llama3", "Hi", &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.text, "test");
    assert!(result.prompt_tokens.is_none());
    assert!(result.completion_tokens.is_none());
    assert!(result.finish_reason.is_none());
}

#[tokio::test]
async fn generate_uses_fallback_model_when_missing() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "response": "ok",
            "done": true
        })))
        .mount(&server)
        .await;

    let provider = OllamaProvider::new(OllamaConfig {
        base_url: server.uri(),
        ..Default::default()
    });

    let result = provider
        .generate("my-model", "Hi", &GenerateOptions::default())
        .await
        .unwrap();

    assert_eq!(result.model, "my-model");
}

// --- Generate: error handling ---

#[tokio::test]
async fn generate_returns_error_on_http_error() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
        .mount(&server)
        .await;

    let provider = OllamaProvider::new(OllamaConfig {
        base_url: server.uri(),
        ..Default::default()
    });

    let result = provider
        .generate("llama3", "Hi", &GenerateOptions::default())
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        LlmError::Http { status, .. } => assert_eq!(status, 500),
        other => panic!("expected Http error, got: {other:?}"),
    }
}

// --- Generate: API key header ---

#[tokio::test]
async fn generate_sends_auth_header_when_api_key_set() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .and(header("Authorization", "Bearer my-secret-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(success_response()))
        .expect(1)
        .mount(&server)
        .await;

    let provider = OllamaProvider::new(OllamaConfig {
        base_url: server.uri(),
        api_key: Some("my-secret-key".to_string()),
        ..Default::default()
    });

    let result = provider
        .generate("llama3", "Hi", &GenerateOptions::default())
        .await;
    assert!(
        result.is_ok(),
        "expected Ok with auth header, got: {result:?}"
    );
}

// --- check_connection ---

#[tokio::test]
async fn check_connection_returns_true_on_200() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": []
        })))
        .mount(&server)
        .await;

    let provider = OllamaProvider::new(OllamaConfig {
        base_url: server.uri(),
        ..Default::default()
    });

    assert!(provider.check_connection().await);
}

#[tokio::test]
async fn check_connection_returns_false_on_error() {
    let provider = OllamaProvider::new(OllamaConfig {
        base_url: "http://127.0.0.1:1".to_string(),
        timeout_secs: 1,
        ..Default::default()
    });

    assert!(!provider.check_connection().await);
}

// --- list_models ---

#[tokio::test]
async fn list_models_parses_model_names() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [
                {"name": "llama3:latest", "size": 1234},
                {"name": "codellama:7b", "size": 5678}
            ]
        })))
        .mount(&server)
        .await;

    let provider = OllamaProvider::new(OllamaConfig {
        base_url: server.uri(),
        ..Default::default()
    });

    let models = provider.list_models().await.unwrap();
    assert_eq!(models, vec!["llama3:latest", "codellama:7b"]);
}

#[tokio::test]
async fn list_models_returns_empty_on_no_models() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": []
        })))
        .mount(&server)
        .await;

    let provider = OllamaProvider::new(OllamaConfig {
        base_url: server.uri(),
        ..Default::default()
    });

    let models = provider.list_models().await.unwrap();
    assert!(models.is_empty());
}

// --- Send + Sync assertions ---

#[test]
fn ollama_provider_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<OllamaProvider>();
}

#[test]
fn ollama_provider_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<OllamaProvider>();
}

#[test]
fn ollama_config_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OllamaConfig>();
}

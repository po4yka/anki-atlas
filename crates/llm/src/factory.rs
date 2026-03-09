use strum::{Display, EnumString};

use crate::error::LlmError;
use crate::ollama::{OllamaConfig, OllamaProvider};
use crate::openrouter::{OpenRouterConfig, OpenRouterProvider};
use crate::provider::LlmProvider;

/// Supported LLM provider types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, Display)]
#[strum(serialize_all = "lowercase")]
pub enum ProviderType {
    Ollama,
    OpenRouter,
}

/// Typed provider configuration.
#[derive(Debug, Clone)]
pub enum ProviderConfig {
    Ollama(OllamaConfig),
    OpenRouter(OpenRouterConfig),
}

/// Create a provider instance from a typed configuration.
pub fn create_provider_from_config(
    config: ProviderConfig,
) -> Result<Box<dyn LlmProvider>, LlmError> {
    match config {
        ProviderConfig::Ollama(c) => Ok(Box::new(OllamaProvider::new(c)?)),
        ProviderConfig::OpenRouter(c) => Ok(Box::new(OpenRouterProvider::new(c)?)),
    }
}

/// Create a provider instance by type from untyped JSON config.
/// For OpenRouter, reads OPENROUTER_API_KEY from env if not in config.
pub fn create_provider(
    provider_type: ProviderType,
    config: serde_json::Value,
) -> Result<Box<dyn LlmProvider>, LlmError> {
    match provider_type {
        ProviderType::Ollama => {
            let defaults = OllamaConfig::default();
            let ollama_config = OllamaConfig {
                base_url: config
                    .get("base_url")
                    .and_then(|v| v.as_str())
                    .map_or(defaults.base_url, String::from),
                timeout_secs: config
                    .get("timeout_secs")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.timeout_secs),
                api_key: config
                    .get("api_key")
                    .and_then(|v| v.as_str())
                    .map(String::from),
            };
            Ok(Box::new(OllamaProvider::new(ollama_config)?))
        }
        ProviderType::OpenRouter => {
            let api_key = config
                .get("api_key")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(String::from)
                .or_else(|| std::env::var("OPENROUTER_API_KEY").ok().filter(|s| !s.is_empty()))
                .ok_or_else(|| LlmError::Provider {
                    message: "OPENROUTER_API_KEY not set and no api_key in config".to_string(),
                    source: None,
                })?;

            let defaults = OpenRouterConfig::default();
            let or_config = OpenRouterConfig {
                api_key,
                base_url: config
                    .get("base_url")
                    .and_then(|v| v.as_str())
                    .map_or(defaults.base_url, String::from),
                timeout_secs: config
                    .get("timeout_secs")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.timeout_secs),
                max_tokens: config
                    .get("max_tokens")
                    .and_then(|v| v.as_u64())
                    .map_or(defaults.max_tokens, |v| v as u32),
                max_retries: config
                    .get("max_retries")
                    .and_then(|v| v.as_u64())
                    .map_or(defaults.max_retries, |v| v as u32),
                site_url: config
                    .get("site_url")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                site_name: config
                    .get("site_name")
                    .and_then(|v| v.as_str())
                    .map(String::from),
            };
            Ok(Box::new(OpenRouterProvider::new(or_config)?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    // --- ProviderType enum tests ---

    #[test]
    fn provider_type_from_str_ollama() {
        let pt = ProviderType::from_str("ollama").unwrap();
        assert_eq!(pt, ProviderType::Ollama);
    }

    #[test]
    fn provider_type_from_str_openrouter() {
        let pt = ProviderType::from_str("openrouter").unwrap();
        assert_eq!(pt, ProviderType::OpenRouter);
    }

    #[test]
    fn provider_type_from_str_invalid() {
        let result = ProviderType::from_str("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn provider_type_display_ollama() {
        assert_eq!(ProviderType::Ollama.to_string(), "ollama");
    }

    #[test]
    fn provider_type_display_openrouter() {
        assert_eq!(ProviderType::OpenRouter.to_string(), "openrouter");
    }

    #[test]
    fn provider_type_debug() {
        let debug = format!("{:?}", ProviderType::Ollama);
        assert!(debug.contains("Ollama"));
    }

    #[test]
    fn provider_type_clone() {
        let pt = ProviderType::Ollama;
        let cloned = pt;
        assert_eq!(pt, cloned);
    }

    #[test]
    fn provider_type_eq() {
        assert_eq!(ProviderType::Ollama, ProviderType::Ollama);
        assert_ne!(ProviderType::Ollama, ProviderType::OpenRouter);
    }

    // --- create_provider tests ---

    #[test]
    fn create_provider_ollama_default_config() {
        let config = serde_json::json!({});
        let result = create_provider(ProviderType::Ollama, config);
        assert!(result.is_ok());
    }

    #[test]
    fn create_provider_ollama_custom_config() {
        let config = serde_json::json!({
            "base_url": "http://myhost:11434",
            "timeout_secs": 120
        });
        let result = create_provider(ProviderType::Ollama, config);
        assert!(result.is_ok());
    }

    #[test]
    fn create_provider_openrouter_with_api_key() {
        let config = serde_json::json!({
            "api_key": "sk-test-key-123"
        });
        let result = create_provider(ProviderType::OpenRouter, config);
        assert!(result.is_ok());
    }

    #[test]
    fn create_provider_openrouter_missing_api_key() {
        let config = serde_json::json!({});
        // Should fail because no api_key in config and no env var set
        let result = create_provider(ProviderType::OpenRouter, config);
        assert!(result.is_err());
    }

    #[test]
    fn create_provider_openrouter_empty_api_key() {
        let config = serde_json::json!({
            "api_key": ""
        });
        let result = create_provider(ProviderType::OpenRouter, config);
        assert!(result.is_err());
    }

    #[test]
    fn create_provider_openrouter_custom_config() {
        let config = serde_json::json!({
            "api_key": "sk-test-key-456",
            "base_url": "https://custom.api.com/v1",
            "timeout_secs": 60,
            "max_tokens": 4096,
            "max_retries": 5,
            "site_url": "https://mysite.com",
            "site_name": "MySite"
        });
        let result = create_provider(ProviderType::OpenRouter, config);
        assert!(result.is_ok());
    }

    #[test]
    fn create_provider_returns_dyn_llm_provider() {
        let config = serde_json::json!({});
        let provider = create_provider(ProviderType::Ollama, config).unwrap();
        // Verify we get a Box<dyn LlmProvider> - this compiles = it works
        let _: Box<dyn LlmProvider> = provider;
    }

    // --- Send + Sync assertions ---

    #[test]
    fn provider_type_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ProviderType>();
    }

    #[test]
    fn provider_type_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ProviderType>();
    }
}

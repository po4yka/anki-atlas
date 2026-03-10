use crate::error::LlmError;
use crate::ollama::{OllamaConfig, OllamaProvider};
use crate::openrouter::{OpenRouterConfig, OpenRouterProvider};
use crate::provider::LlmProvider;

/// Typed provider configuration.
#[derive(Debug, Clone)]
pub enum ProviderConfig {
    Ollama(OllamaConfig),
    OpenRouter(OpenRouterConfig),
}

/// Create a provider instance from typed configuration.
pub fn create_provider(config: ProviderConfig) -> Result<Box<dyn LlmProvider>, LlmError> {
    match config {
        ProviderConfig::Ollama(config) => Ok(Box::new(OllamaProvider::new(config)?)),
        ProviderConfig::OpenRouter(config) => Ok(Box::new(OpenRouterProvider::new(config)?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_provider_ollama_default_config() {
        let result = create_provider(ProviderConfig::Ollama(OllamaConfig::default()));
        assert!(result.is_ok());
    }

    #[test]
    fn create_provider_ollama_custom_config() {
        let result = create_provider(ProviderConfig::Ollama(OllamaConfig {
            base_url: "http://myhost:11434".to_string(),
            timeout_secs: 120,
            api_key: Some("token".to_string()),
        }));
        assert!(result.is_ok());
    }

    #[test]
    fn create_provider_openrouter_with_api_key() {
        let result = create_provider(ProviderConfig::OpenRouter(OpenRouterConfig {
            api_key: "sk-test-key-123".to_string(),
            ..OpenRouterConfig::default()
        }));
        assert!(result.is_ok());
    }

    #[test]
    fn create_provider_openrouter_missing_api_key() {
        let result = create_provider(ProviderConfig::OpenRouter(OpenRouterConfig::default()));
        assert!(result.is_err());
    }

    #[test]
    fn create_provider_openrouter_empty_api_key() {
        let result = create_provider(ProviderConfig::OpenRouter(OpenRouterConfig {
            api_key: String::new(),
            ..OpenRouterConfig::default()
        }));
        assert!(result.is_err());
    }

    #[test]
    fn create_provider_openrouter_custom_config() {
        let result = create_provider(ProviderConfig::OpenRouter(OpenRouterConfig {
            api_key: "sk-test-key-456".to_string(),
            base_url: "https://custom.api.com/v1".to_string(),
            timeout_secs: 60,
            max_tokens: 4096,
            max_retries: 5,
            site_url: Some("https://mysite.com".to_string()),
            site_name: Some("MySite".to_string()),
        }));
        assert!(result.is_ok());
    }

    #[test]
    fn create_provider_returns_dyn_llm_provider() {
        let provider = create_provider(ProviderConfig::Ollama(OllamaConfig::default())).unwrap();
        let _: Box<dyn LlmProvider> = provider;
    }
}

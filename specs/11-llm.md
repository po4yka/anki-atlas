# Spec: crate `llm`

## Source Reference
Python: `packages/llm/` (base.py, factory.py, openrouter.py, ollama.py)

## Purpose
Async LLM provider abstraction with trait-based dispatch, retry logic, and JSON-mode support. Provides a unified interface for calling OpenRouter (cloud) and Ollama (local) LLM APIs via reqwest. The factory selects a provider by enum variant. All types are `Send + Sync` so providers can be shared across Tokio tasks behind `Arc`.

## Dependencies
```toml
[dependencies]
common = { path = "../common" }
async-trait = "0.1"
reqwest = { version = "0.12", features = ["json", "rustls-tls"], default-features = false }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
tokio = { version = "1", features = ["time"] }
tracing = "0.1"
strum = { version = "0.27", features = ["derive"] }

[dev-dependencies]
mockall = "0.13"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
wiremock = "0.6"
```

## Public API

### Response (`src/response.rs`)

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Structured response from an LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub text: String,
    pub model: String,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: Option<u32>,
    pub finish_reason: Option<String>,
    pub raw: HashMap<String, serde_json::Value>,
}

impl LlmResponse {
    /// Total tokens used (prompt + completion), if both are known.
    pub fn total_tokens(&self) -> Option<u32> {
        match (self.prompt_tokens, self.completion_tokens) {
            (Some(p), Some(c)) => Some(p + c),
            _ => None,
        }
    }
}
```

### Error (`src/error.rs`)

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("provider error: {message}")]
    Provider { message: String, source: Option<Box<dyn std::error::Error + Send + Sync>> },

    #[error("invalid JSON response: {message}")]
    InvalidJson { message: String, response_text: String },

    #[error("request failed after {attempts} retries: {message}")]
    Exhausted { message: String, attempts: u32 },

    #[error("HTTP error {status}: {body}")]
    Http { status: u16, body: String },

    #[error("connection error: {0}")]
    Connection(String),
}
```

### Provider Trait (`src/provider.rs`)

```rust
use async_trait::async_trait;
use std::collections::HashMap;

use crate::error::LlmError;
use crate::response::LlmResponse;

/// Options for a generation request.
#[derive(Debug, Clone, Default)]
pub struct GenerateOptions {
    pub system: String,
    pub temperature: f32,        // default 0.7
    pub json_mode: bool,
    pub json_schema: Option<serde_json::Value>,
}

/// Trait for LLM providers. All implementations must be Send + Sync.
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait LlmProvider: Send + Sync {
    /// Generate a completion.
    async fn generate(
        &self,
        model: &str,
        prompt: &str,
        opts: &GenerateOptions,
    ) -> Result<LlmResponse, LlmError>;

    /// Generate and parse JSON output. Default implementation calls generate
    /// with json_mode=true and deserializes.
    async fn generate_json(
        &self,
        model: &str,
        prompt: &str,
        opts: &GenerateOptions,
    ) -> Result<HashMap<String, serde_json::Value>, LlmError> {
        let mut json_opts = opts.clone();
        json_opts.json_mode = true;
        let response = self.generate(model, prompt, &json_opts).await?;
        let parsed: HashMap<String, serde_json::Value> =
            serde_json::from_str(&response.text).map_err(|e| LlmError::InvalidJson {
                message: format!("LLM returned invalid JSON: {e}"),
                response_text: response.text[..response.text.len().min(500)].to_string(),
            })?;
        Ok(parsed)
    }

    /// Check if the provider API is reachable.
    async fn check_connection(&self) -> bool;

    /// List available model identifiers.
    async fn list_models(&self) -> Result<Vec<String>, LlmError>;
}
```

### OpenRouter Provider (`src/openrouter.rs`)

```rust
use crate::provider::{GenerateOptions, LlmProvider};
use crate::response::LlmResponse;
use crate::error::LlmError;

/// OpenRouter provider configuration.
#[derive(Debug, Clone)]
pub struct OpenRouterConfig {
    pub api_key: String,
    pub base_url: String,       // default "https://openrouter.ai/api/v1"
    pub timeout_secs: u64,      // default 180
    pub max_tokens: u32,        // default 2048
    pub max_retries: u32,       // default 3
    pub site_url: Option<String>,
    pub site_name: Option<String>,
}

pub struct OpenRouterProvider {
    config: OpenRouterConfig,
    client: reqwest::Client,
}

impl OpenRouterProvider {
    /// Create a new provider. Returns LlmError::Provider if api_key is empty.
    pub fn new(config: OpenRouterConfig) -> Result<Self, LlmError>;
}

// #[async_trait] impl LlmProvider for OpenRouterProvider { ... }
```

### Ollama Provider (`src/ollama.rs`)

```rust
use crate::provider::{GenerateOptions, LlmProvider};

/// Ollama provider configuration.
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub base_url: String,       // default "http://localhost:11434"
    pub api_key: Option<String>,
    pub timeout_secs: u64,      // default 300
}

pub struct OllamaProvider {
    config: OllamaConfig,
    client: reqwest::Client,
}

impl OllamaProvider {
    pub fn new(config: OllamaConfig) -> Self;
}

// #[async_trait] impl LlmProvider for OllamaProvider { ... }
```

### Factory (`src/factory.rs`)

```rust
use strum::{Display, EnumString};
use crate::provider::LlmProvider;
use crate::error::LlmError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, Display)]
#[strum(serialize_all = "lowercase")]
pub enum ProviderType {
    Ollama,
    OpenRouter,
}

/// Create a provider instance by type.
/// For OpenRouter, reads OPENROUTER_API_KEY from env if not in config.
pub fn create_provider(
    provider_type: ProviderType,
    config: serde_json::Value,
) -> Result<Box<dyn LlmProvider>, LlmError>;
```

### Module root (`src/lib.rs`)

```rust
pub mod error;
pub mod factory;
pub mod ollama;
pub mod openrouter;
pub mod provider;
pub mod response;

pub use error::LlmError;
pub use factory::{create_provider, ProviderType};
pub use provider::{GenerateOptions, LlmProvider};
pub use response::LlmResponse;
```

## Internal Details

### Retry Logic (OpenRouter)
- Retryable HTTP status codes: 429, 502, 503, 504.
- Exponential backoff: `2^(attempt+1)` seconds, up to `max_retries` (default 3).
- Non-retryable errors (e.g., 400, 401) fail immediately.
- Uses `tokio::time::sleep` for async delay between retries.

### OpenRouter Request Format
- Endpoint: `POST {base_url}/chat/completions`
- Messages array with optional system message + user message.
- `response_format` set to `{"type": "json_schema", ...}` when `json_schema` is provided, or `{"type": "json_object"}` when `json_mode` is true.
- Response parsing: extract `choices[0].message.content`, `usage.prompt_tokens`, `usage.completion_tokens`, `choices[0].finish_reason`.

### Ollama Request Format
- Endpoint: `POST {base_url}/api/generate`
- Payload: `model`, `prompt`, `stream: false`, `system` (if non-empty), `options.temperature`.
- `format: "json"` when `json_mode` is true.
- Response: `response`, `model`, `prompt_eval_count`, `eval_count`, `done_reason`.
- Health check: `GET {base_url}/api/tags`
- List models: `GET {base_url}/api/tags` -> `models[].name`

### Thread Safety
- `reqwest::Client` is internally `Arc`-ed and safe to clone.
- Provider structs hold only `config` (Clone) + `client`, making them `Send + Sync`.
- Providers should be wrapped in `Arc<dyn LlmProvider>` for sharing across tasks.

## Acceptance Criteria
- [ ] `LlmResponse::total_tokens()` returns `Some` only when both fields are `Some`
- [ ] `MockLlmProvider` compiles and can be used in downstream crate tests
- [ ] `OpenRouterProvider::new` returns error when API key is empty
- [ ] OpenRouter retries on 429/502/503/504 with exponential backoff (wiremock test)
- [ ] OpenRouter fails immediately on 400/401 (wiremock test)
- [ ] OpenRouter `generate` correctly builds messages array with/without system prompt
- [ ] OpenRouter `generate` sets `response_format` for json_mode and json_schema
- [ ] OpenRouter `_parse_response` extracts text, tokens, finish_reason from valid response
- [ ] OpenRouter `_parse_response` returns error on empty choices
- [ ] Ollama `generate` builds correct payload with system prompt and temperature
- [ ] Ollama `generate` sets `format: "json"` when json_mode is true
- [ ] Ollama `check_connection` returns true on 200, false on network error
- [ ] Ollama `list_models` parses model names from `/api/tags` response
- [ ] `create_provider` dispatches correctly for both ProviderType variants
- [ ] `create_provider` returns error for missing API key on OpenRouter
- [ ] `generate_json` default implementation parses valid JSON and rejects non-object
- [ ] All provider types are `Send + Sync` (compile-time assertion)
- [ ] `make check` equivalent passes (clippy, fmt, test)

use serde::Deserialize;

/// Qdrant quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Quantization {
    None,
    Scalar,
    Binary,
}

/// Configuration error returned by Settings::load() and Settings::validate().
#[derive(Debug)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Application settings loaded from env vars prefixed `ANKIATLAS_`.
#[derive(Debug, Clone, Deserialize)]
pub struct Settings {
    pub postgres_url: String,
    pub qdrant_url: String,
    pub qdrant_quantization: Quantization,
    pub qdrant_on_disk: bool,
    pub redis_url: String,
    pub job_queue_name: String,
    pub job_result_ttl_seconds: u32,
    pub job_max_retries: u32,
    pub embedding_provider: String,
    pub embedding_model: String,
    pub embedding_dimension: u32,
    pub rerank_enabled: bool,
    pub rerank_model: String,
    pub rerank_top_n: u32,
    pub rerank_batch_size: u32,
    pub api_host: String,
    pub api_port: u16,
    pub api_key: Option<String>,
    pub debug: bool,
    pub anki_collection_path: Option<String>,
}

impl Settings {
    /// Load settings from environment variables and optional `.env` file.
    /// Validates all fields after loading.
    pub fn load() -> Result<Self, ConfigError> {
        Err(ConfigError("TODO: not implemented".to_string()))
    }

    /// Validate all fields. Called automatically by `load()`.
    pub fn validate(&self) -> Result<(), ConfigError> {
        Err(ConfigError("TODO: not implemented".to_string()))
    }
}

/// Return a lazily-initialized, globally cached `&'static Settings`.
pub fn get_settings() -> &'static Settings {
    todo!("not implemented")
}

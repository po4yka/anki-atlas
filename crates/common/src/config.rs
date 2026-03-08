use std::env;
use std::sync::OnceLock;

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
        let settings = Self {
            postgres_url: env_or(
                "ANKIATLAS_POSTGRES_URL",
                "postgresql://localhost:5432/ankiatlas",
            ),
            qdrant_url: env_or("ANKIATLAS_QDRANT_URL", "http://localhost:6333"),
            qdrant_quantization: env_or("ANKIATLAS_QDRANT_QUANTIZATION", "scalar")
                .parse_quantization()?,
            qdrant_on_disk: env_or("ANKIATLAS_QDRANT_ON_DISK", "false")
                .parse_bool("qdrant_on_disk")?,
            redis_url: env_or("ANKIATLAS_REDIS_URL", "redis://localhost:6379/0"),
            job_queue_name: env_or("ANKIATLAS_JOB_QUEUE_NAME", "ankiatlas_jobs"),
            job_result_ttl_seconds: env_or("ANKIATLAS_JOB_RESULT_TTL_SECONDS", "86400")
                .parse_u32("job_result_ttl_seconds")?,
            job_max_retries: env_or("ANKIATLAS_JOB_MAX_RETRIES", "3")
                .parse_u32("job_max_retries")?,
            embedding_provider: env_or("ANKIATLAS_EMBEDDING_PROVIDER", "openai"),
            embedding_model: env_or("ANKIATLAS_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimension: env_or("ANKIATLAS_EMBEDDING_DIMENSION", "1536")
                .parse_u32("embedding_dimension")?,
            rerank_enabled: env_or("ANKIATLAS_RERANK_ENABLED", "false")
                .parse_bool("rerank_enabled")?,
            rerank_model: env_or(
                "ANKIATLAS_RERANK_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ),
            rerank_top_n: env_or("ANKIATLAS_RERANK_TOP_N", "50").parse_u32("rerank_top_n")?,
            rerank_batch_size: env_or("ANKIATLAS_RERANK_BATCH_SIZE", "32")
                .parse_u32("rerank_batch_size")?,
            api_host: env_or("ANKIATLAS_API_HOST", "0.0.0.0"),
            api_port: env_or("ANKIATLAS_API_PORT", "8000")
                .parse::<u16>()
                .map_err(|e| ConfigError(format!("invalid api_port: {e}")))?,
            api_key: env::var("ANKIATLAS_API_KEY").ok().filter(|s| !s.is_empty()),
            debug: env_or("ANKIATLAS_DEBUG", "false").parse_bool("debug")?,
            anki_collection_path: env::var("ANKIATLAS_ANKI_COLLECTION_PATH")
                .ok()
                .filter(|s| !s.is_empty()),
        };

        settings.validate()?;
        Ok(settings)
    }

    /// Validate all fields. Called automatically by `load()`.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // postgres_url must start with postgresql:// or postgres://
        if !self.postgres_url.starts_with("postgresql://")
            && !self.postgres_url.starts_with("postgres://")
        {
            return Err(ConfigError(format!(
                "postgres_url must start with postgresql:// or postgres://, got: {}",
                self.postgres_url
            )));
        }

        // qdrant_url must start with http:// or https://
        if !self.qdrant_url.starts_with("http://") && !self.qdrant_url.starts_with("https://") {
            return Err(ConfigError(format!(
                "qdrant_url must start with http:// or https://, got: {}",
                self.qdrant_url
            )));
        }

        // redis_url must start with redis:// or rediss://
        if !self.redis_url.starts_with("redis://") && !self.redis_url.starts_with("rediss://") {
            return Err(ConfigError(format!(
                "redis_url must start with redis:// or rediss://, got: {}",
                self.redis_url
            )));
        }

        // embedding_dimension must be positive and in valid set (unless mock provider)
        if self.embedding_dimension == 0 {
            return Err(ConfigError(
                "embedding_dimension must be positive".to_string(),
            ));
        }
        if self.embedding_provider != "mock" {
            const VALID_DIMS: [u32; 5] = [384, 768, 1024, 1536, 3072];
            if !VALID_DIMS.contains(&self.embedding_dimension) {
                return Err(ConfigError(format!(
                    "embedding_dimension {} not in valid set: {VALID_DIMS:?}",
                    self.embedding_dimension
                )));
            }
        }

        // Positive integer fields
        if self.job_result_ttl_seconds == 0 {
            return Err(ConfigError(
                "job_result_ttl_seconds must be positive".to_string(),
            ));
        }
        if self.job_max_retries == 0 {
            return Err(ConfigError("job_max_retries must be positive".to_string()));
        }
        if self.rerank_top_n == 0 {
            return Err(ConfigError("rerank_top_n must be positive".to_string()));
        }
        if self.rerank_batch_size == 0 {
            return Err(ConfigError(
                "rerank_batch_size must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

/// Return a lazily-initialized, globally cached `&'static Settings`.
pub fn get_settings() -> &'static Settings {
    static SETTINGS: OnceLock<Settings> = OnceLock::new();
    SETTINGS.get_or_init(|| Settings::load().expect("failed to load settings"))
}

fn env_or(key: &str, default: &str) -> String {
    env::var(key).unwrap_or_else(|_| default.to_string())
}

trait ParseHelper {
    fn parse_u32(self, field: &str) -> Result<u32, ConfigError>;
    fn parse_bool(self, field: &str) -> Result<bool, ConfigError>;
    fn parse_quantization(self) -> Result<Quantization, ConfigError>;
}

impl ParseHelper for String {
    fn parse_u32(self, field: &str) -> Result<u32, ConfigError> {
        self.parse::<u32>()
            .map_err(|e| ConfigError(format!("invalid {field}: {e}")))
    }

    fn parse_bool(self, field: &str) -> Result<bool, ConfigError> {
        self.parse::<bool>()
            .map_err(|e| ConfigError(format!("invalid {field}: {e}")))
    }

    fn parse_quantization(self) -> Result<Quantization, ConfigError> {
        match self.as_str() {
            "none" => Ok(Quantization::None),
            "scalar" => Ok(Quantization::Scalar),
            "binary" => Ok(Quantization::Binary),
            other => Err(ConfigError(format!(
                "invalid quantization: {other}, expected none|scalar|binary"
            ))),
        }
    }
}

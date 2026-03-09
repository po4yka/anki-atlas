use crate::error::JobError;

/// Redis connection configuration parsed from URL.
#[derive(Debug, Clone)]
pub struct RedisConfig {
    pub host: String,
    pub port: u16,
    pub database: u32,
    pub username: Option<String>,
    pub password: Option<String>,
    pub tls: bool,
}

/// Parse a Redis URL (redis:// or rediss://) into config.
pub fn parse_redis_url(_redis_url: &str) -> Result<RedisConfig, JobError> {
    // TODO(impl): implement
    Err(JobError::Redis("not implemented".to_string()))
}

/// Create a rustis Client from a Redis URL.
pub async fn create_redis_client(_redis_url: &str) -> Result<rustis::client::Client, JobError> {
    // TODO(impl): implement
    Err(JobError::BackendUnavailable("not implemented".to_string()))
}

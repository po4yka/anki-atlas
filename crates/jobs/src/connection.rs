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
pub fn parse_redis_url(redis_url: &str) -> Result<RedisConfig, JobError> {
    let parsed =
        url::Url::parse(redis_url).map_err(|e| JobError::Redis(format!("invalid URL: {e}")))?;

    let tls = match parsed.scheme() {
        "redis" => false,
        "rediss" => true,
        other => return Err(JobError::Redis(format!("unsupported scheme: {other}"))),
    };

    let host = parsed.host_str().unwrap_or("localhost").to_string();
    let port = parsed.port().unwrap_or(6379);

    let database = parsed
        .path()
        .trim_start_matches('/')
        .parse::<u32>()
        .unwrap_or(0);

    let username = if !parsed.username().is_empty() {
        Some(parsed.username().to_string())
    } else {
        None
    };

    let password = parsed.password().map(|p| p.to_string());

    Ok(RedisConfig {
        host,
        port,
        database,
        username,
        password,
        tls,
    })
}

/// Create a rustis Client from a Redis URL.
pub async fn create_redis_client(redis_url: &str) -> Result<rustis::client::Client, JobError> {
    rustis::client::Client::connect(redis_url)
        .await
        .map_err(|e| JobError::Redis(format!("connection failed: {e}")))
}

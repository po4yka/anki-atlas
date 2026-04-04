use super::*;
use std::time::Duration;

/// Helper to build a Settings with defaults for testing.
fn test_settings() -> common::config::Settings {
    common::config::Settings {
        postgres_url: "postgresql://localhost:5432/ankiatlas".to_string(),
        qdrant_url: "http://localhost:6333".to_string(),
        qdrant_quantization: common::config::Quantization::Scalar,
        qdrant_on_disk: false,
        postgres_url: "postgres://localhost:5432/anki_atlas".to_string(),
        job_queue_name: "ankiatlas_jobs".to_string(),
        job_result_ttl_seconds: 86400,
        job_max_retries: 3,
        embedding_provider: common::config::EmbeddingProviderKind::Mock,
        embedding_model: "test".to_string(),
        embedding_dimension: 384,
        rerank_enabled: false,
        rerank_model: "test".to_string(),
        rerank_top_n: 50,
        rerank_batch_size: 32,
        api_host: "0.0.0.0".to_string(),
        api_port: 8000,
        api_key: None,
        debug: false,
        anki_collection_path: None,
        anki_media_root: None,
    }
}

#[test]
fn from_settings_maps_postgres_url() {
    let settings = test_settings();
    let config = WorkerConfig::from_job_settings(&settings.jobs());
    assert_eq!(config.postgres_url, "redis://localhost:6379/0");
}

#[test]
fn from_settings_maps_queue_name() {
    let settings = test_settings();
    let config = WorkerConfig::from_job_settings(&settings.jobs());
    assert_eq!(config.queue_name, "ankiatlas_jobs");
}

#[test]
fn from_settings_default_max_concurrency() {
    let settings = test_settings();
    let config = WorkerConfig::from_job_settings(&settings.jobs());
    assert_eq!(config.max_concurrency, 4);
}

#[test]
fn from_settings_maps_max_retries() {
    let settings = test_settings();
    let config = WorkerConfig::from_job_settings(&settings.jobs());
    assert_eq!(config.max_retries, 3);
}

#[test]
fn from_settings_default_poll_interval() {
    let settings = test_settings();
    let config = WorkerConfig::from_job_settings(&settings.jobs());
    assert_eq!(config.poll_interval, Duration::from_secs(1));
}

#[test]
fn from_settings_default_allow_abort_on_shutdown() {
    let settings = test_settings();
    let config = WorkerConfig::from_job_settings(&settings.jobs());
    assert!(config.allow_abort_on_shutdown);
}

#[test]
fn from_settings_maps_result_ttl() {
    let settings = test_settings();
    let config = WorkerConfig::from_job_settings(&settings.jobs());
    assert_eq!(config.result_ttl_seconds, 86400);
}

#[test]
fn from_settings_custom_max_retries() {
    let mut settings = test_settings();
    settings.job_max_retries = 5;
    let config = WorkerConfig::from_job_settings(&settings.jobs());
    assert_eq!(config.max_retries, 5);
}

#[test]
fn from_settings_custom_postgres_url() {
    let mut settings = test_settings();
    settings.postgres_url = "postgres://custom:5432/test".to_string();
    let config = WorkerConfig::from_job_settings(&settings.jobs());
    assert_eq!(config.postgres_url, "postgres://custom:5432/test");
}

#[test]
fn from_settings_custom_result_ttl() {
    let mut settings = test_settings();
    settings.job_result_ttl_seconds = 3600;
    let config = WorkerConfig::from_job_settings(&settings.jobs());
    assert_eq!(config.result_ttl_seconds, 3600);
}

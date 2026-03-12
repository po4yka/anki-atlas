use common::config::*;

// ── Send + Sync ──────────────────────────────────────────────────────────

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn settings_and_quantization_are_send_and_sync() {
    assert_send_sync::<Settings>();
    assert_send_sync::<Quantization>();
    assert_send_sync::<EmbeddingProviderKind>();
}

// ── Quantization enum ───────────────────────────────────────────────────

#[test]
fn quantization_serde_lowercase() {
    let q: Quantization = serde_json::from_str(r#""scalar""#).unwrap();
    assert_eq!(q, Quantization::Scalar);

    let q: Quantization = serde_json::from_str(r#""binary""#).unwrap();
    assert_eq!(q, Quantization::Binary);

    let q: Quantization = serde_json::from_str(r#""none""#).unwrap();
    assert_eq!(q, Quantization::None);
}

#[test]
fn quantization_debug() {
    let q = Quantization::Scalar;
    let debug = format!("{q:?}");
    assert!(
        debug.contains("Scalar"),
        "Debug should contain variant name"
    );
}

// ── Settings::load() defaults ───────────────────────────────────────────

#[test]
fn settings_load_returns_defaults_when_no_env_vars() {
    // Clear all ANKIATLAS_ env vars, then load should succeed with defaults
    temp_env::with_vars_unset(
        vec![
            "ANKIATLAS_POSTGRES_URL",
            "ANKIATLAS_QDRANT_URL",
            "ANKIATLAS_REDIS_URL",
            "ANKIATLAS_EMBEDDING_PROVIDER",
            "ANKIATLAS_EMBEDDING_MODEL",
            "ANKIATLAS_EMBEDDING_DIMENSION",
            "ANKIATLAS_QDRANT_QUANTIZATION",
            "ANKIATLAS_QDRANT_ON_DISK",
            "ANKIATLAS_JOB_QUEUE_NAME",
            "ANKIATLAS_JOB_RESULT_TTL_SECONDS",
            "ANKIATLAS_JOB_MAX_RETRIES",
            "ANKIATLAS_RERANK_ENABLED",
            "ANKIATLAS_RERANK_MODEL",
            "ANKIATLAS_RERANK_TOP_N",
            "ANKIATLAS_RERANK_BATCH_SIZE",
            "ANKIATLAS_API_HOST",
            "ANKIATLAS_API_PORT",
            "ANKIATLAS_API_KEY",
            "ANKIATLAS_DEBUG",
            "ANKIATLAS_ANKI_COLLECTION_PATH",
            "ANKIATLAS_ANKI_MEDIA_ROOT",
        ],
        || {
            let settings = Settings::load().expect("load with defaults should succeed");

            // Database
            assert!(settings.postgres_url.starts_with("postgresql://"));

            // Vector store
            assert!(settings.qdrant_url.starts_with("http://"));
            assert_eq!(settings.qdrant_quantization, Quantization::Scalar);
            assert!(!settings.qdrant_on_disk);

            // Async jobs
            assert!(settings.redis_url.starts_with("redis://"));
            assert_eq!(settings.job_queue_name, "ankiatlas_jobs");
            assert_eq!(settings.job_result_ttl_seconds, 86400);
            assert_eq!(settings.job_max_retries, 3);

            // Embeddings
            assert_eq!(settings.embedding_provider, EmbeddingProviderKind::OpenAi);
            assert_eq!(settings.embedding_model, "text-embedding-3-small");
            assert_eq!(settings.embedding_dimension, 1536);
            assert!(!settings.rerank_enabled);
            assert!(!settings.rerank_model.is_empty());
            assert_eq!(settings.rerank_top_n, 50);
            assert_eq!(settings.rerank_batch_size, 32);

            // API
            assert_eq!(settings.api_host, "0.0.0.0");
            assert_eq!(settings.api_port, 8000);
            assert!(settings.api_key.is_none());
            assert!(!settings.debug);

            // Anki source
            assert!(settings.anki_collection_path.is_none());
            assert!(settings.anki_media_root.is_none());
        },
    );
}

// ── Settings::load() from env vars ──────────────────────────────────────

#[test]
fn settings_load_reads_ankiatlas_prefixed_env_vars() {
    temp_env::with_vars(
        vec![
            (
                "ANKIATLAS_POSTGRES_URL",
                Some("postgresql://custom:1234/db"),
            ),
            ("ANKIATLAS_QDRANT_URL", Some("https://qdrant.example.com")),
            (
                "ANKIATLAS_REDIS_URL",
                Some("redis://redis.example.com:6379/1"),
            ),
            ("ANKIATLAS_API_PORT", Some("9090")),
            ("ANKIATLAS_DEBUG", Some("true")),
            ("ANKIATLAS_API_KEY", Some("secret-key-123")),
            (
                "ANKIATLAS_ANKI_COLLECTION_PATH",
                Some("/path/to/collection.anki2"),
            ),
            (
                "ANKIATLAS_ANKI_MEDIA_ROOT",
                Some("/path/to/collection.media"),
            ),
            ("ANKIATLAS_EMBEDDING_DIMENSION", Some("768")),
        ],
        || {
            let settings = Settings::load().expect("load with env vars should succeed");

            assert_eq!(settings.postgres_url, "postgresql://custom:1234/db");
            assert_eq!(settings.qdrant_url, "https://qdrant.example.com");
            assert_eq!(settings.redis_url, "redis://redis.example.com:6379/1");
            assert_eq!(settings.api_port, 9090);
            assert!(settings.debug);
            assert_eq!(settings.api_key, Some("secret-key-123".to_string()));
            assert_eq!(
                settings.anki_collection_path,
                Some("/path/to/collection.anki2".to_string())
            );
            assert_eq!(
                settings.anki_media_root,
                Some("/path/to/collection.media".to_string())
            );
            assert_eq!(settings.embedding_dimension, 768);
            assert_eq!(settings.embedding_provider, EmbeddingProviderKind::OpenAi);
        },
    );
}

// ── Validation: postgres_url ────────────────────────────────────────────

#[test]
fn validate_rejects_invalid_postgres_url() {
    temp_env::with_vars(
        vec![("ANKIATLAS_POSTGRES_URL", Some("mysql://localhost/db"))],
        || {
            let result = Settings::load();
            assert!(result.is_err(), "Should reject non-postgresql URL");
        },
    );
}

#[test]
fn validate_accepts_postgres_scheme() {
    temp_env::with_vars(
        vec![("ANKIATLAS_POSTGRES_URL", Some("postgres://localhost/db"))],
        || {
            let settings = Settings::load().expect("postgres:// should be accepted");
            assert!(settings.postgres_url.starts_with("postgres://"));
        },
    );
}

// ── Validation: qdrant_url ──────────────────────────────────────────────

#[test]
fn validate_rejects_invalid_qdrant_url() {
    temp_env::with_vars(
        vec![("ANKIATLAS_QDRANT_URL", Some("grpc://localhost:6334"))],
        || {
            let result = Settings::load();
            assert!(result.is_err(), "Should reject non-http(s) URL");
        },
    );
}

#[test]
fn validate_accepts_https_qdrant_url() {
    temp_env::with_vars(
        vec![("ANKIATLAS_QDRANT_URL", Some("https://qdrant.cloud:6333"))],
        || {
            let settings = Settings::load().expect("https:// qdrant URL should be accepted");
            assert_eq!(settings.qdrant_url, "https://qdrant.cloud:6333");
        },
    );
}

#[test]
fn qdrant_grpc_url_swaps_default_rest_port() {
    let grpc_url = qdrant_grpc_url("http://localhost:6333").expect("grpc url");
    assert_eq!(grpc_url, "http://localhost:6334");
}

#[test]
fn qdrant_grpc_url_preserves_custom_port() {
    let grpc_url = qdrant_grpc_url("http://localhost:7444").expect("grpc url");
    assert_eq!(grpc_url, "http://localhost:7444");
}

// ── Validation: redis_url ───────────────────────────────────────────────

#[test]
fn validate_rejects_invalid_redis_url() {
    temp_env::with_vars(
        vec![("ANKIATLAS_REDIS_URL", Some("http://localhost:6379"))],
        || {
            let result = Settings::load();
            assert!(result.is_err(), "Should reject non-redis URL");
        },
    );
}

#[test]
fn validate_accepts_rediss_url() {
    temp_env::with_vars(
        vec![("ANKIATLAS_REDIS_URL", Some("rediss://secure-redis:6380/0"))],
        || {
            let settings = Settings::load().expect("rediss:// should be accepted");
            assert!(settings.redis_url.starts_with("rediss://"));
        },
    );
}

// ── Validation: embedding_dimension ─────────────────────────────────────

#[test]
fn validate_rejects_invalid_embedding_dimension() {
    temp_env::with_vars(vec![("ANKIATLAS_EMBEDDING_DIMENSION", Some("512"))], || {
        let result = Settings::load();
        assert!(
            result.is_err(),
            "Should reject dimension 512 (not in valid set)"
        );
    });
}

#[test]
fn validate_rejects_zero_embedding_dimension() {
    temp_env::with_vars(vec![("ANKIATLAS_EMBEDDING_DIMENSION", Some("0"))], || {
        let result = Settings::load();
        assert!(result.is_err(), "Should reject dimension 0");
    });
}

#[test]
fn validate_accepts_all_valid_embedding_dimensions() {
    for dim in [384, 768, 1024, 1536, 3072] {
        temp_env::with_vars(
            vec![("ANKIATLAS_EMBEDDING_DIMENSION", Some(&dim.to_string()))],
            || {
                let settings = Settings::load()
                    .unwrap_or_else(|_| panic!("dimension {dim} should be accepted"));
                assert_eq!(settings.embedding_dimension, dim);
            },
        );
    }
}

#[test]
fn validate_accepts_gemini_embedding_2_custom_dimension_up_to_3072() {
    temp_env::with_vars(
        vec![
            ("ANKIATLAS_EMBEDDING_PROVIDER", Some("google")),
            (
                "ANKIATLAS_EMBEDDING_MODEL",
                Some("gemini-embedding-2-preview"),
            ),
            ("ANKIATLAS_EMBEDDING_DIMENSION", Some("2048")),
        ],
        || {
            let settings = Settings::load().expect("Gemini Embedding 2 should accept 2048");
            assert_eq!(settings.embedding_dimension, 2048);
        },
    );
}

#[test]
fn validate_rejects_gemini_embedding_2_dimension_above_3072() {
    temp_env::with_vars(
        vec![
            ("ANKIATLAS_EMBEDDING_PROVIDER", Some("google")),
            (
                "ANKIATLAS_EMBEDDING_MODEL",
                Some("gemini-embedding-2-preview"),
            ),
            ("ANKIATLAS_EMBEDDING_DIMENSION", Some("4096")),
        ],
        || {
            let result = Settings::load();
            assert!(result.is_err(), "Gemini Embedding 2 should reject 4096");
        },
    );
}

#[test]
fn validate_mock_provider_accepts_any_positive_dimension() {
    temp_env::with_vars(
        vec![
            ("ANKIATLAS_EMBEDDING_PROVIDER", Some("mock")),
            ("ANKIATLAS_EMBEDDING_DIMENSION", Some("42")),
        ],
        || {
            let settings = Settings::load().expect("mock provider should accept any positive dim");
            assert_eq!(settings.embedding_dimension, 42);
            assert_eq!(settings.embedding_provider, EmbeddingProviderKind::Mock);
        },
    );
}

#[test]
fn validate_rejects_unknown_embedding_provider() {
    temp_env::with_vars(
        vec![("ANKIATLAS_EMBEDDING_PROVIDER", Some("mystery"))],
        || {
            let result = Settings::load();
            assert!(result.is_err(), "Should reject unknown embedding provider");
        },
    );
}

// ── Validation: positive integer fields ─────────────────────────────────

#[test]
fn validate_rejects_zero_job_result_ttl() {
    temp_env::with_vars(
        vec![("ANKIATLAS_JOB_RESULT_TTL_SECONDS", Some("0"))],
        || {
            let result = Settings::load();
            assert!(result.is_err(), "Should reject job_result_ttl_seconds = 0");
        },
    );
}

#[test]
fn validate_rejects_zero_job_max_retries() {
    temp_env::with_vars(vec![("ANKIATLAS_JOB_MAX_RETRIES", Some("0"))], || {
        let result = Settings::load();
        assert!(result.is_err(), "Should reject job_max_retries = 0");
    });
}

#[test]
fn validate_rejects_zero_rerank_top_n() {
    temp_env::with_vars(vec![("ANKIATLAS_RERANK_TOP_N", Some("0"))], || {
        let result = Settings::load();
        assert!(result.is_err(), "Should reject rerank_top_n = 0");
    });
}

#[test]
fn validate_rejects_zero_rerank_batch_size() {
    temp_env::with_vars(vec![("ANKIATLAS_RERANK_BATCH_SIZE", Some("0"))], || {
        let result = Settings::load();
        assert!(result.is_err(), "Should reject rerank_batch_size = 0");
    });
}

// ── ConfigError type ────────────────────────────────────────────────────

#[test]
fn settings_load_returns_config_error_variant() {
    temp_env::with_vars(
        vec![("ANKIATLAS_POSTGRES_URL", Some("invalid://url"))],
        || {
            let err = Settings::load().unwrap_err();
            // ConfigError should be the error type returned by Settings::load()
            let msg = format!("{err}");
            assert!(!msg.is_empty(), "ConfigError should have a display message");
        },
    );
}

// ── Debug impl ──────────────────────────────────────────────────────────

#[test]
fn settings_debug_impl() {
    temp_env::with_vars_unset(
        vec![
            "ANKIATLAS_POSTGRES_URL",
            "ANKIATLAS_QDRANT_URL",
            "ANKIATLAS_REDIS_URL",
        ],
        || {
            let settings = Settings::load().expect("defaults should work");
            let debug = format!("{settings:?}");
            assert!(debug.contains("Settings"), "Debug should contain type name");
        },
    );
}

// ── Clone impl ──────────────────────────────────────────────────────────

#[test]
fn settings_clone() {
    temp_env::with_vars_unset(
        vec![
            "ANKIATLAS_POSTGRES_URL",
            "ANKIATLAS_QDRANT_URL",
            "ANKIATLAS_REDIS_URL",
        ],
        || {
            let settings = Settings::load().expect("defaults should work");
            let cloned = settings.clone();
            assert_eq!(settings.postgres_url, cloned.postgres_url);
            assert_eq!(settings.api_port, cloned.api_port);
        },
    );
}

// ── Crate root re-exports ───────────────────────────────────────────────

#[test]
fn crate_root_reexports_settings_types() {
    let _: fn() -> std::result::Result<common::Settings, _> = common::Settings::load;
    let _: common::EmbeddingProviderKind = common::EmbeddingProviderKind::Mock;
}

// ── Quantization default ────────────────────────────────────────────────

#[test]
fn quantization_clone_copy_eq() {
    let q = Quantization::Scalar;
    let q2 = q; // Copy
    let q3 = q.clone(); // Clone
    assert_eq!(q, q2);
    assert_eq!(q, q3);
}

// ── Case insensitive env vars ───────────────────────────────────────────

#[test]
fn settings_load_qdrant_quantization_from_env() {
    temp_env::with_vars(
        vec![("ANKIATLAS_QDRANT_QUANTIZATION", Some("binary"))],
        || {
            let settings = Settings::load().expect("should load with quantization override");
            assert_eq!(settings.qdrant_quantization, Quantization::Binary);
        },
    );
}

#[test]
fn settings_projection_methods_return_narrow_runtime_contracts() {
    temp_env::with_vars_unset(
        vec![
            "ANKIATLAS_POSTGRES_URL",
            "ANKIATLAS_QDRANT_URL",
            "ANKIATLAS_REDIS_URL",
            "ANKIATLAS_API_KEY",
        ],
        || {
            let settings = Settings::load().expect("defaults should work");

            let database = settings.database();
            assert!(database.postgres_url.starts_with("postgresql://"));

            let jobs = settings.jobs();
            assert!(jobs.redis_url.starts_with("redis://"));
            assert_eq!(jobs.queue_name, "ankiatlas_jobs");

            let api = settings.api();
            assert_eq!(api.host, "0.0.0.0");
            assert_eq!(api.port, 8000);

            let embedding = settings.embedding();
            assert_eq!(embedding.provider, EmbeddingProviderKind::OpenAi);
            assert_eq!(embedding.dimension, 1536);

            let rerank = settings.rerank();
            assert_eq!(rerank.top_n, 50);
            assert_eq!(rerank.batch_size, 32);
        },
    );
}

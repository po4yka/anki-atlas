use indexer::embeddings::{
    EmbeddingError, EmbeddingProvider, EmbeddingProviderConfig, MockEmbeddingProvider,
    content_hash, create_embedding_provider,
};

// ── MockEmbeddingProvider ──────────────────────────────────────────

#[test]
fn mock_provider_model_name() {
    let provider = MockEmbeddingProvider::new(384);
    assert_eq!(provider.model_name(), "mock/test");
}

#[test]
fn mock_provider_dimension() {
    let provider = MockEmbeddingProvider::new(384);
    assert_eq!(provider.dimension(), 384);
}

#[tokio::test]
async fn mock_provider_embed_returns_correct_count() {
    let provider = MockEmbeddingProvider::new(384);
    let texts = vec!["hello".to_string(), "world".to_string()];
    let result = provider.embed(&texts).await.unwrap();
    assert_eq!(result.len(), 2);
}

#[tokio::test]
async fn mock_provider_embed_returns_correct_dimension() {
    let provider = MockEmbeddingProvider::new(128);
    let texts = vec!["test".to_string()];
    let result = provider.embed(&texts).await.unwrap();
    assert_eq!(result[0].len(), 128);
}

#[tokio::test]
async fn mock_provider_embed_is_deterministic() {
    let provider = MockEmbeddingProvider::new(64);
    let texts = vec!["deterministic".to_string()];
    let result1 = provider.embed(&texts).await.unwrap();
    let result2 = provider.embed(&texts).await.unwrap();
    assert_eq!(result1, result2);
}

#[tokio::test]
async fn mock_provider_different_texts_produce_different_vectors() {
    let provider = MockEmbeddingProvider::new(64);
    let result1 = provider.embed(&["hello".to_string()]).await.unwrap();
    let result2 = provider.embed(&["world".to_string()]).await.unwrap();
    assert_ne!(result1[0], result2[0]);
}

#[tokio::test]
async fn mock_provider_values_in_range() {
    let provider = MockEmbeddingProvider::new(384);
    let result = provider.embed(&["range test".to_string()]).await.unwrap();
    for val in &result[0] {
        assert!(
            *val >= -1.0 && *val <= 1.0,
            "value {val} out of [-1.0, 1.0]"
        );
    }
}

#[tokio::test]
async fn mock_provider_empty_batch_returns_empty() {
    let provider = MockEmbeddingProvider::new(64);
    let result = provider.embed(&[]).await.unwrap();
    assert!(result.is_empty());
}

// ── content_hash ──────────────────────────────────────────────────

#[test]
fn content_hash_length_is_16_hex_chars() {
    let hash = content_hash("mock/test", "hello");
    assert_eq!(hash.len(), 16);
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn content_hash_includes_model_name() {
    let hash_a = content_hash("model_a", "same text");
    let hash_b = content_hash("model_b", "same text");
    assert_ne!(
        hash_a, hash_b,
        "different models must produce different hashes"
    );
}

#[test]
fn content_hash_is_deterministic() {
    let hash1 = content_hash("mock/test", "test");
    let hash2 = content_hash("mock/test", "test");
    assert_eq!(hash1, hash2);
}

#[test]
fn content_hash_different_text_different_hash() {
    let hash_a = content_hash("mock/test", "aaa");
    let hash_b = content_hash("mock/test", "bbb");
    assert_ne!(hash_a, hash_b);
}

// ── EmbeddingProviderConfig deserialization ────────────────────────

#[test]
fn config_deserialize_mock() {
    let json = r#"{"type": "mock", "dimension": 128}"#;
    let config: EmbeddingProviderConfig = serde_json::from_str(json).unwrap();
    match config {
        EmbeddingProviderConfig::Mock { dimension } => assert_eq!(dimension, 128),
        _ => panic!("expected Mock variant"),
    }
}

#[test]
fn config_deserialize_openai() {
    let json = r#"{"type": "open_ai", "model": "text-embedding-3-small", "dimension": 1536, "batch_size": 50}"#;
    let config: EmbeddingProviderConfig = serde_json::from_str(json).unwrap();
    match config {
        EmbeddingProviderConfig::OpenAi {
            model,
            dimension,
            batch_size,
        } => {
            assert_eq!(model, "text-embedding-3-small");
            assert_eq!(dimension, 1536);
            assert_eq!(batch_size, Some(50));
        }
        _ => panic!("expected OpenAi variant"),
    }
}

#[test]
fn config_deserialize_google() {
    let json = r#"{"type": "google", "model": "text-embedding-004", "dimension": 768}"#;
    let config: EmbeddingProviderConfig = serde_json::from_str(json).unwrap();
    match config {
        EmbeddingProviderConfig::Google {
            model,
            dimension,
            batch_size,
        } => {
            assert_eq!(model, "text-embedding-004");
            assert_eq!(dimension, 768);
            assert_eq!(batch_size, None);
        }
        _ => panic!("expected Google variant"),
    }
}

// ── create_embedding_provider factory ─────────────────────────────

#[test]
fn factory_creates_mock_provider() {
    let config = EmbeddingProviderConfig::Mock { dimension: 64 };
    let provider = create_embedding_provider(&config).unwrap();
    assert_eq!(provider.model_name(), "mock/test");
    assert_eq!(provider.dimension(), 64);
}

#[tokio::test]
async fn factory_mock_provider_embeds_correctly() {
    let config = EmbeddingProviderConfig::Mock { dimension: 32 };
    let provider = create_embedding_provider(&config).unwrap();
    let result = provider.embed(&["test".to_string()]).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].len(), 32);
}

// ── Send + Sync bounds ────────────────────────────────────────────

#[test]
fn mock_provider_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MockEmbeddingProvider>();
}

#[test]
fn embedding_error_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<EmbeddingError>();
}

// ── Edge cases ────────────────────────────────────────────────────

#[test]
fn content_hash_empty_text() {
    let hash = content_hash("mock/test", "");
    assert_eq!(hash.len(), 16);
}

#[tokio::test]
async fn mock_provider_large_batch() {
    let provider = MockEmbeddingProvider::new(16);
    let texts: Vec<String> = (0..100).map(|i| format!("text_{i}")).collect();
    let result = provider.embed(&texts).await.unwrap();
    assert_eq!(result.len(), 100);
    for vec in &result {
        assert_eq!(vec.len(), 16);
    }
}

#[tokio::test]
async fn mock_provider_dimension_1() {
    let provider = MockEmbeddingProvider::new(1);
    let result = provider.embed(&["x".to_string()]).await.unwrap();
    assert_eq!(result[0].len(), 1);
}

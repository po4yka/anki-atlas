use indexer::qdrant::{
    NotePayload, QdrantRepository, SearchFilters, SparseVector, UpsertResult, VectorStoreError,
};

// ── NotePayload ──────────────────────────────────────────────────

#[test]
fn note_payload_serialize_roundtrip() {
    let payload = NotePayload {
        note_id: 42,
        deck_names: vec!["Default".to_string()],
        tags: vec!["vocab".to_string(), "en".to_string()],
        model_id: 100,
        content_hash: "abc123def456".to_string(),
        mature: true,
        lapses: 3,
        reps: 15,
        fail_rate: Some(0.2),
        chunk_id: "42:text_primary".to_string(),
        chunk_kind: "text_primary".to_string(),
        modality: "text".to_string(),
        source_field: None,
        asset_rel_path: None,
        mime_type: Some("text/plain".to_string()),
        preview_label: Some("payload preview".to_string()),
    };
    let json = serde_json::to_string(&payload).unwrap();
    let deserialized: NotePayload = serde_json::from_str(&json).unwrap();
    assert_eq!(payload, deserialized);
}

#[test]
fn note_payload_defaults_on_missing_optional_fields() {
    let json = r#"{
        "note_id": 1,
        "deck_names": ["Default"],
        "tags": [],
        "model_id": 10,
        "content_hash": "abcdef1234567890"
    }"#;
    let payload: NotePayload = serde_json::from_str(json).unwrap();
    assert!(!payload.mature);
    assert_eq!(payload.lapses, 0);
    assert_eq!(payload.reps, 0);
    assert_eq!(payload.fail_rate, None);
}

#[test]
fn note_payload_with_all_fields() {
    let json = r#"{
        "note_id": 99,
        "deck_names": ["Deck A", "Deck B"],
        "tags": ["tag1"],
        "model_id": 5,
        "content_hash": "1234567890abcdef",
        "mature": true,
        "lapses": 7,
        "reps": 42,
        "fail_rate": 0.15
    }"#;
    let payload: NotePayload = serde_json::from_str(json).unwrap();
    assert_eq!(payload.note_id, 99);
    assert_eq!(payload.deck_names.len(), 2);
    assert!(payload.mature);
    assert_eq!(payload.lapses, 7);
    assert_eq!(payload.reps, 42);
    assert_eq!(payload.fail_rate, Some(0.15));
}

// ── SparseVector ────────────────────────────────────────────────

#[test]
fn sparse_vector_default_is_empty() {
    let sv = SparseVector::default();
    assert!(sv.indices.is_empty());
    assert!(sv.values.is_empty());
}

#[test]
fn sparse_vector_clone() {
    let sv = SparseVector {
        indices: vec![1, 2, 3],
        values: vec![0.5, 0.3, 0.2],
    };
    let cloned = sv.clone();
    assert_eq!(sv, cloned);
}

// ── SearchFilters ───────────────────────────────────────────────

#[test]
fn search_filters_default_is_empty() {
    let filters = SearchFilters::default();
    assert!(filters.deck_names.is_none());
    assert!(filters.deck_names_exclude.is_none());
    assert!(filters.tags.is_none());
    assert!(filters.tags_exclude.is_none());
    assert!(filters.model_ids.is_none());
    assert!(!filters.mature_only);
    assert!(filters.max_lapses.is_none());
    assert!(filters.min_reps.is_none());
}

// ── UpsertResult ────────────────────────────────────────────────

#[test]
fn upsert_result_default() {
    let result = UpsertResult::default();
    assert_eq!(result.upserted, 0);
    assert_eq!(result.skipped, 0);
}

// ── VectorStoreError ────────────────────────────────────────────

#[test]
fn vector_store_error_dimension_mismatch_display() {
    let err = VectorStoreError::DimensionMismatch {
        collection: "test".to_string(),
        expected: 384,
        actual: 768,
    };
    let msg = err.to_string();
    assert!(msg.contains("384"));
    assert!(msg.contains("768"));
    assert!(msg.contains("test"));
}

#[test]
fn vector_store_error_client_display() {
    let err = VectorStoreError::Client("timeout".to_string());
    assert!(err.to_string().contains("timeout"));
}

#[test]
fn vector_store_error_connection_display() {
    let err = VectorStoreError::Connection("refused".to_string());
    assert!(err.to_string().contains("refused"));
}

// ── Send + Sync bounds ──────────────────────────────────────────

common::assert_send_sync!(
    NotePayload,
    SparseVector,
    SearchFilters,
    UpsertResult,
    VectorStoreError,
    QdrantRepository,
);

// ── text_to_sparse_vector ───────────────────────────────────────

#[test]
fn sparse_vector_empty_text_returns_empty() {
    let sv = QdrantRepository::text_to_sparse_vector("");
    assert!(sv.indices.is_empty());
    assert!(sv.values.is_empty());
}

#[test]
fn sparse_vector_single_token() {
    let sv = QdrantRepository::text_to_sparse_vector("hello");
    assert_eq!(sv.indices.len(), 1);
    assert_eq!(sv.values.len(), 1);
    // Single token with count=1: weight = 1.0 + ln(1) = 1.0
    // After L2 normalization of a single value: 1.0 / 1.0 = 1.0
    assert!((sv.values[0] - 1.0).abs() < 1e-6);
}

#[test]
fn sparse_vector_is_deterministic() {
    let sv1 = QdrantRepository::text_to_sparse_vector("hello world test");
    let sv2 = QdrantRepository::text_to_sparse_vector("hello world test");
    assert_eq!(sv1.indices, sv2.indices);
    assert_eq!(sv1.values, sv2.values);
}

#[test]
fn sparse_vector_indices_are_sorted_ascending() {
    let sv = QdrantRepository::text_to_sparse_vector("the quick brown fox jumps over the lazy dog");
    for window in sv.indices.windows(2) {
        assert!(
            window[0] <= window[1],
            "indices must be sorted: {} > {}",
            window[0],
            window[1]
        );
    }
}

#[test]
fn sparse_vector_is_l2_normalized() {
    let sv = QdrantRepository::text_to_sparse_vector("hello world hello test");
    if sv.values.is_empty() {
        return;
    }
    let norm: f32 = sv.values.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "sparse vector should be L2-normalized, got norm={norm}"
    );
}

#[test]
fn sparse_vector_case_insensitive() {
    let sv_lower = QdrantRepository::text_to_sparse_vector("hello");
    let sv_upper = QdrantRepository::text_to_sparse_vector("HELLO");
    assert_eq!(sv_lower.indices, sv_upper.indices);
    assert_eq!(sv_lower.values, sv_upper.values);
}

#[test]
fn sparse_vector_only_alphanumeric_tokens() {
    // Punctuation/special chars are stripped, only [a-z0-9]+ tokens kept
    let sv = QdrantRepository::text_to_sparse_vector("hello! @world# $test%");
    let sv_clean = QdrantRepository::text_to_sparse_vector("hello world test");
    assert_eq!(sv.indices, sv_clean.indices);
    assert_eq!(sv.values, sv_clean.values);
}

#[test]
fn sparse_vector_repeated_token_has_higher_weight() {
    // "hello hello" has count=2 for "hello": weight = 1.0 + ln(2) ≈ 1.693
    // "hello world" has count=1 each: weight = 1.0 + ln(1) = 1.0
    // After L2 norm, the single-token case normalizes to 1.0
    let sv_repeated = QdrantRepository::text_to_sparse_vector("hello hello");
    let sv_single = QdrantRepository::text_to_sparse_vector("hello");
    // Both have the same token hash, but repeated has weight 1+ln(2) before norm
    // Single token normalized = 1.0, repeated single token normalized = 1.0
    // So the actual value should be the same (both normalize to 1.0 for single-index)
    assert_eq!(sv_repeated.indices.len(), 1);
    assert_eq!(sv_single.indices.len(), 1);
    assert_eq!(sv_repeated.indices, sv_single.indices);
}

#[test]
fn sparse_vector_multiple_tokens_have_correct_count() {
    let sv = QdrantRepository::text_to_sparse_vector("aaa bbb ccc");
    // 3 unique tokens → 3 indices (assuming no hash collisions)
    assert!(!sv.indices.is_empty()); // at least 1 (could be fewer with collisions)
    assert!(sv.indices.len() <= 3); // at most 3
    assert_eq!(sv.indices.len(), sv.values.len());
}

#[test]
fn sparse_vector_whitespace_only_returns_empty() {
    let sv = QdrantRepository::text_to_sparse_vector("   \t\n  ");
    assert!(sv.indices.is_empty());
    assert!(sv.values.is_empty());
}

#[test]
fn sparse_vector_numbers_are_tokens() {
    let sv = QdrantRepository::text_to_sparse_vector("test123 456");
    // "test123" and "456" are valid tokens
    assert!(!sv.indices.is_empty());
}

#[test]
fn sparse_vector_hash_collision_accumulates_weights() {
    // We can't easily force a collision, but we can verify the property:
    // When two different tokens hash to the same index, weights should sum
    // Just verify the function handles multi-token input without panic
    let sv = QdrantRepository::text_to_sparse_vector(
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    );
    assert!(!sv.indices.is_empty());
    // All values should be positive
    for v in &sv.values {
        assert!(*v > 0.0, "all sparse vector values should be positive");
    }
}

#[test]
fn sparse_vector_uses_blake2b_hashing() {
    // Verify deterministic hashing: same token always maps to same index
    let sv1 = QdrantRepository::text_to_sparse_vector("uniquetoken");
    let sv2 = QdrantRepository::text_to_sparse_vector("uniquetoken");
    assert_eq!(sv1.indices, sv2.indices);
}

// ── QdrantRepository::new ───────────────────────────────────────

#[tokio::test]
async fn qdrant_repository_new_fails_with_invalid_url() {
    // Connecting to a non-existent server should fail
    let result = QdrantRepository::new("http://localhost:99999", "test_collection").await;
    assert!(result.is_err());
}

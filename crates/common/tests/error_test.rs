use common::error::*;
use std::collections::HashMap;
use std::error::Error;

// ── Send + Sync ──────────────────────────────────────────────────────────

common::assert_send_sync!(AnkiAtlasError);

// ── Display / Error trait ────────────────────────────────────────────────

#[test]
fn database_connection_display() {
    let err = AnkiAtlasError::DatabaseConnection {
        message: "timeout".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "database connection failed: timeout");
    // Ensure std::error::Error is implemented
    let _: &dyn Error = &err;
}

#[test]
fn migration_display() {
    let err = AnkiAtlasError::Migration {
        message: "v3 failed".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "migration failed: v3 failed");
}

#[test]
fn vector_store_connection_display() {
    let err = AnkiAtlasError::VectorStoreConnection {
        message: "refused".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "vector store connection failed: refused");
}

#[test]
fn collection_display() {
    let err = AnkiAtlasError::Collection {
        message: "create failed".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(
        err.to_string(),
        "collection operation failed: create failed"
    );
}

#[test]
fn dimension_mismatch_display() {
    let err = AnkiAtlasError::DimensionMismatch {
        collection: "notes".to_string(),
        expected: 1536,
        actual: 768,
    };
    assert_eq!(
        err.to_string(),
        "dimension mismatch on 'notes': expected 1536, got 768"
    );
}

#[test]
fn embedding_display() {
    let err = AnkiAtlasError::Embedding {
        message: "failed".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "embedding error: failed");
}

#[test]
fn embedding_api_display() {
    let err = AnkiAtlasError::EmbeddingApi {
        message: "401".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "embedding API error: 401");
}

#[test]
fn embedding_timeout_display() {
    let err = AnkiAtlasError::EmbeddingTimeout {
        message: "30s".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "embedding timeout: 30s");
}

#[test]
fn embedding_model_changed_display() {
    let err = AnkiAtlasError::EmbeddingModelChanged {
        stored: "text-embedding-ada-002".to_string(),
        current: "text-embedding-3-small".to_string(),
    };
    assert_eq!(
        err.to_string(),
        "embedding model changed: 'text-embedding-ada-002' -> 'text-embedding-3-small'. Use --force-reindex."
    );
}

#[test]
fn sync_display() {
    let err = AnkiAtlasError::Sync {
        message: "failed".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "sync error: failed");
}

#[test]
fn collection_not_found_display() {
    let err = AnkiAtlasError::CollectionNotFound {
        message: "missing".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "collection not found: missing");
}

#[test]
fn sync_conflict_display() {
    let err = AnkiAtlasError::SyncConflict {
        message: "version mismatch".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "sync conflict: version mismatch");
}

#[test]
fn anki_connect_display() {
    let err = AnkiAtlasError::AnkiConnect {
        message: "refused".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "AnkiConnect error: refused");
}

#[test]
fn anki_reader_display() {
    let err = AnkiAtlasError::AnkiReader {
        message: "corrupt".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "Anki reader error: corrupt");
}

#[test]
fn configuration_display() {
    let err = AnkiAtlasError::Configuration {
        message: "missing key".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "configuration error: missing key");
}

#[test]
fn not_found_display() {
    let err = AnkiAtlasError::NotFound {
        message: "card 42".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "not found: card 42");
}

#[test]
fn conflict_display() {
    let err = AnkiAtlasError::Conflict {
        message: "duplicate".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "conflict: duplicate");
}

#[test]
fn card_generation_display() {
    let err = AnkiAtlasError::CardGeneration {
        message: "LLM failed".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "card generation error: LLM failed");
}

#[test]
fn card_validation_display() {
    let err = AnkiAtlasError::CardValidation {
        message: "empty front".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "card validation error: empty front");
}

#[test]
fn provider_display() {
    let err = AnkiAtlasError::Provider {
        message: "rate limited".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "provider error: rate limited");
}

#[test]
fn obsidian_parse_display() {
    let err = AnkiAtlasError::ObsidianParse {
        message: "bad yaml".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "obsidian parse error: bad yaml");
}

#[test]
fn job_backend_unavailable_display() {
    let err = AnkiAtlasError::JobBackendUnavailable {
        message: "redis down".to_string(),
        context: HashMap::new(),
    };
    assert_eq!(err.to_string(), "job backend unavailable: redis down");
}

// ── ErrorContext ─────────────────────────────────────────────────────────

#[test]
fn error_carries_context() {
    let mut ctx = HashMap::new();
    ctx.insert("host".to_string(), "localhost".to_string());
    ctx.insert("port".to_string(), "5432".to_string());

    let err = AnkiAtlasError::DatabaseConnection {
        message: "refused".to_string(),
        context: ctx,
    };

    // Context is accessible but not in Display
    assert!(!err.to_string().contains("host"));
    if let AnkiAtlasError::DatabaseConnection { context, .. } = &err {
        assert_eq!(context.get("host").unwrap(), "localhost");
        assert_eq!(context.get("port").unwrap(), "5432");
    } else {
        panic!("wrong variant");
    }
}

#[test]
fn error_empty_context() {
    let err = AnkiAtlasError::Embedding {
        message: "fail".to_string(),
        context: HashMap::new(),
    };
    if let AnkiAtlasError::Embedding { context, .. } = &err {
        assert!(context.is_empty());
    }
}

// ── Result type alias ───────────────────────────────────────────────────

#[test]
fn result_alias_works() {
    fn ok_fn() -> Result<i32> {
        Ok(42)
    }
    fn err_fn() -> Result<i32> {
        Err(AnkiAtlasError::NotFound {
            message: "nope".to_string(),
            context: HashMap::new(),
        })
    }
    assert!(ok_fn().is_ok());
    assert!(err_fn().is_err());
}

// ── WithContext trait ────────────────────────────────────────────────────

#[test]
fn with_context_adds_key_value() {
    let err = AnkiAtlasError::DatabaseConnection {
        message: "timeout".to_string(),
        context: HashMap::new(),
    };
    let err = err.with_context("host", "db.example.com");
    if let AnkiAtlasError::DatabaseConnection { context, .. } = &err {
        assert_eq!(context.get("host").unwrap(), "db.example.com");
    } else {
        panic!("wrong variant");
    }
}

#[test]
fn with_context_chains() {
    let err = AnkiAtlasError::Embedding {
        message: "fail".to_string(),
        context: HashMap::new(),
    }
    .with_context("model", "ada-002")
    .with_context("provider", "openai");

    if let AnkiAtlasError::Embedding { context, .. } = &err {
        assert_eq!(context.get("model").unwrap(), "ada-002");
        assert_eq!(context.get("provider").unwrap(), "openai");
    } else {
        panic!("wrong variant");
    }
}

#[test]
fn with_context_on_variant_without_context_is_noop() {
    // DimensionMismatch has no ErrorContext field
    let err = AnkiAtlasError::DimensionMismatch {
        collection: "notes".to_string(),
        expected: 1536,
        actual: 768,
    };
    // Should return self unchanged (no panic)
    let err = err.with_context("key", "value");
    assert_eq!(
        err.to_string(),
        "dimension mismatch on 'notes': expected 1536, got 768"
    );
}

#[test]
fn with_context_on_embedding_model_changed_is_noop() {
    let err = AnkiAtlasError::EmbeddingModelChanged {
        stored: "old".to_string(),
        current: "new".to_string(),
    };
    let err = err.with_context("key", "value");
    assert!(err.to_string().contains("old"));
}

// ── Debug impl ──────────────────────────────────────────────────────────

#[test]
fn all_variants_implement_debug() {
    let _ = format!(
        "{:?}",
        AnkiAtlasError::DatabaseConnection {
            message: "x".to_string(),
            context: HashMap::new(),
        }
    );
    let _ = format!(
        "{:?}",
        AnkiAtlasError::DimensionMismatch {
            collection: "c".to_string(),
            expected: 1,
            actual: 2,
        }
    );
    let _ = format!(
        "{:?}",
        AnkiAtlasError::EmbeddingModelChanged {
            stored: "a".to_string(),
            current: "b".to_string(),
        }
    );
}

// ── Re-exports from crate root ──────────────────────────────────────────

#[test]
fn crate_root_reexports_error_types() {
    // These should be accessible from common::AnkiAtlasError and common::Result
    let _: common::Result<()> = Ok(());
    let _err = common::AnkiAtlasError::NotFound {
        message: "test".to_string(),
        context: HashMap::new(),
    };
}

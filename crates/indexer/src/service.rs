use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::instrument;

use crate::embeddings::{self, EmbeddingProvider};
use crate::qdrant::{NotePayload, QdrantRepository, VectorRepository};

/// A note prepared for indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteForIndexing {
    pub note_id: i64,
    pub model_id: i64,
    pub normalized_text: String,
    pub tags: Vec<String>,
    pub deck_names: Vec<String>,
    #[serde(default)]
    pub mature: bool,
    #[serde(default)]
    pub lapses: i32,
    #[serde(default)]
    pub reps: i32,
    #[serde(default)]
    pub fail_rate: Option<f64>,
}

/// Statistics from an indexing operation.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct IndexStats {
    pub notes_processed: usize,
    pub notes_embedded: usize,
    pub notes_skipped: usize,
    pub notes_deleted: usize,
    pub errors: Vec<String>,
}

/// Index service errors.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("embedding failed: {0}")]
    Embedding(#[from] crate::embeddings::EmbeddingError),
    #[error("vector store: {0}")]
    VectorStore(#[from] crate::qdrant::VectorStoreError),
    #[error("database: {0}")]
    Database(String),
    #[error("embedding model changed: stored={stored}, current={current}")]
    ModelChanged { stored: String, current: String },
}

/// Index service. Generic over dependencies for testability.
pub struct IndexService<E: EmbeddingProvider, V: VectorRepository> {
    embedding: Arc<E>,
    vector_repo: Arc<V>,
}

impl<E: EmbeddingProvider, V: VectorRepository> IndexService<E, V> {
    pub fn new(embedding: E, vector_repo: V) -> Self {
        Self {
            embedding: Arc::new(embedding),
            vector_repo: Arc::new(vector_repo),
        }
    }

    /// Index a batch of notes. Skips notes whose content_hash is unchanged
    /// unless `force_reindex` is true.
    #[instrument(skip(self, notes), fields(note_count = notes.len()))]
    pub async fn index_notes(
        &self,
        notes: &[NoteForIndexing],
        force_reindex: bool,
    ) -> Result<IndexStats, IndexError> {
        if notes.is_empty() {
            return Ok(IndexStats::default());
        }

        let note_ids: Vec<i64> = notes.iter().map(|n| n.note_id).collect();
        let existing_hashes = self.vector_repo.get_existing_hashes(&note_ids).await?;

        let model_name = self.embedding.model_name();

        // Compute hashes once and determine which notes need embedding
        let mut to_embed: Vec<(&NoteForIndexing, String)> = Vec::new();
        let mut skipped = 0usize;

        for note in notes {
            let new_hash = embeddings::content_hash(model_name, &note.normalized_text);
            if !force_reindex {
                if let Some(existing) = existing_hashes.get(&note.note_id) {
                    if *existing == new_hash {
                        skipped += 1;
                        continue;
                    }
                }
            }
            to_embed.push((note, new_hash));
        }

        let mut stats = IndexStats {
            notes_processed: notes.len(),
            notes_skipped: skipped,
            ..Default::default()
        };

        if to_embed.is_empty() {
            return Ok(stats);
        }

        // Embed texts
        let texts: Vec<String> = to_embed
            .iter()
            .map(|(n, _)| n.normalized_text.clone())
            .collect();
        let vectors = self.embedding.embed(&texts).await?;

        // Build payloads using cached hashes
        let payloads: Vec<NotePayload> = to_embed
            .iter()
            .map(|(n, hash)| NotePayload {
                note_id: n.note_id,
                model_id: n.model_id,
                deck_names: n.deck_names.clone(),
                tags: n.tags.clone(),
                content_hash: hash.clone(),
                mature: n.mature,
                lapses: n.lapses,
                reps: n.reps,
                fail_rate: n.fail_rate,
            })
            .collect();

        // Generate sparse vectors
        let sparse_vectors: Vec<_> = to_embed
            .iter()
            .map(|(n, _)| QdrantRepository::text_to_sparse_vector(&n.normalized_text))
            .collect();

        let upserted = self
            .vector_repo
            .upsert_vectors(&vectors, &payloads, Some(&sparse_vectors))
            .await?;

        stats.notes_embedded = upserted;

        Ok(stats)
    }

    /// Delete notes from the vector store by ID.
    #[instrument(skip(self), fields(note_count = note_ids.len()))]
    pub async fn delete_notes(&self, note_ids: &[i64]) -> Result<usize, IndexError> {
        Ok(self.vector_repo.delete_vectors(note_ids).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::{EmbeddingError, MockEmbeddingProvider};
    use crate::qdrant::{MockVectorRepository, VectorStoreError};
    use std::collections::HashMap;

    // -- Helper to build a NoteForIndexing with defaults --

    fn make_note(note_id: i64, text: &str) -> NoteForIndexing {
        NoteForIndexing {
            note_id,
            model_id: 1,
            normalized_text: text.to_string(),
            tags: vec!["tag1".to_string()],
            deck_names: vec!["Default".to_string()],
            mature: false,
            lapses: 0,
            reps: 0,
            fail_rate: None,
        }
    }

    // ====================================================================
    // Data type tests
    // ====================================================================

    #[test]
    fn note_for_indexing_serde_roundtrip() {
        let note = make_note(42, "hello world");
        let json = serde_json::to_string(&note).unwrap();
        let deserialized: NoteForIndexing = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.note_id, 42);
        assert_eq!(deserialized.normalized_text, "hello world");
    }

    #[test]
    fn note_for_indexing_defaults() {
        let json = r#"{"note_id":1,"model_id":1,"normalized_text":"x","tags":[],"deck_names":[]}"#;
        let note: NoteForIndexing = serde_json::from_str(json).unwrap();
        assert!(!note.mature);
        assert_eq!(note.lapses, 0);
        assert_eq!(note.reps, 0);
        assert!(note.fail_rate.is_none());
    }

    #[test]
    fn index_stats_default() {
        let stats = IndexStats::default();
        assert_eq!(stats.notes_processed, 0);
        assert_eq!(stats.notes_embedded, 0);
        assert_eq!(stats.notes_skipped, 0);
        assert_eq!(stats.notes_deleted, 0);
        assert!(stats.errors.is_empty());
    }

    #[test]
    fn index_stats_serializable() {
        let stats = IndexStats {
            notes_processed: 10,
            notes_embedded: 8,
            notes_skipped: 2,
            notes_deleted: 0,
            errors: vec!["err1".to_string()],
        };
        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("\"notes_processed\":10"));
        assert!(json.contains("\"errors\":[\"err1\"]"));
    }

    #[test]
    fn index_error_from_embedding_error() {
        let err = EmbeddingError::NotConfigured("test".into());
        let idx_err: IndexError = err.into();
        assert!(matches!(idx_err, IndexError::Embedding(_)));
        assert!(idx_err.to_string().contains("test"));
    }

    #[test]
    fn index_error_from_vector_store_error() {
        let err = VectorStoreError::Client("qdrant down".into());
        let idx_err: IndexError = err.into();
        assert!(matches!(idx_err, IndexError::VectorStore(_)));
        assert!(idx_err.to_string().contains("qdrant down"));
    }

    #[test]
    fn index_error_model_changed_message() {
        let err = IndexError::ModelChanged {
            stored: "old-model".into(),
            current: "new-model".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("old-model"));
        assert!(msg.contains("new-model"));
    }

    #[test]
    fn index_error_database() {
        let err = IndexError::Database("connection refused".into());
        assert!(err.to_string().contains("connection refused"));
    }

    // ====================================================================
    // Send + Sync bounds
    // ====================================================================

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    #[test]
    fn note_for_indexing_is_send_sync() {
        assert_send::<NoteForIndexing>();
        assert_sync::<NoteForIndexing>();
    }

    #[test]
    fn index_stats_is_send_sync() {
        assert_send::<IndexStats>();
        assert_sync::<IndexStats>();
    }

    #[test]
    fn index_error_is_send_sync() {
        assert_send::<IndexError>();
        assert_sync::<IndexError>();
    }

    #[test]
    fn index_service_is_send_sync() {
        assert_send::<IndexService<MockEmbeddingProvider, MockVectorRepository>>();
        assert_sync::<IndexService<MockEmbeddingProvider, MockVectorRepository>>();
    }

    // ====================================================================
    // IndexService::new
    // ====================================================================

    #[test]
    fn service_new_constructs() {
        let embedding = MockEmbeddingProvider::new(128);
        let repo = MockVectorRepository::new();
        let _service = IndexService::new(embedding, repo);
    }

    // ====================================================================
    // IndexService::index_notes - hash skip behavior
    // ====================================================================

    #[tokio::test]
    async fn index_notes_empty_input_returns_zero_stats() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        repo.expect_get_existing_hashes()
            .returning(|_| Box::pin(async { Ok(HashMap::new()) }));

        let service = IndexService::new(embedding, repo);
        let stats = service.index_notes(&[], false).await.unwrap();

        assert_eq!(stats.notes_processed, 0);
        assert_eq!(stats.notes_embedded, 0);
        assert_eq!(stats.notes_skipped, 0);
    }

    #[tokio::test]
    async fn index_notes_all_new_embeds_all() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        repo.expect_get_existing_hashes()
            .returning(|_| Box::pin(async { Ok(HashMap::new()) }));

        repo.expect_upsert_vectors()
            .withf(|vectors, payloads, _sparse| vectors.len() == 2 && payloads.len() == 2)
            .returning(|vectors, _, _| {
                let len = vectors.len();
                Box::pin(async move { Ok(len) })
            });

        let service = IndexService::new(embedding, repo);
        let notes = vec![make_note(1, "hello"), make_note(2, "world")];
        let stats = service.index_notes(&notes, false).await.unwrap();

        assert_eq!(stats.notes_processed, 2);
        assert_eq!(stats.notes_embedded, 2);
        assert_eq!(stats.notes_skipped, 0);
    }

    #[tokio::test]
    async fn index_notes_skips_unchanged_hashes() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        let hash = crate::embeddings::content_hash("mock/test", "hello");
        let mut existing = HashMap::new();
        existing.insert(1_i64, hash);

        repo.expect_get_existing_hashes().returning(move |_| {
            let e = existing.clone();
            Box::pin(async move { Ok(e) })
        });

        repo.expect_upsert_vectors()
            .withf(|vectors, payloads, _| vectors.len() == 1 && payloads[0].note_id == 2)
            .returning(|vectors, _, _| {
                let len = vectors.len();
                Box::pin(async move { Ok(len) })
            });

        let service = IndexService::new(embedding, repo);
        let notes = vec![make_note(1, "hello"), make_note(2, "world")];
        let stats = service.index_notes(&notes, false).await.unwrap();

        assert_eq!(stats.notes_processed, 2);
        assert_eq!(stats.notes_embedded, 1);
        assert_eq!(stats.notes_skipped, 1);
    }

    #[tokio::test]
    async fn index_notes_force_reindex_embeds_all_even_with_matching_hashes() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        let hash = crate::embeddings::content_hash("mock/test", "hello");
        let mut existing = HashMap::new();
        existing.insert(1_i64, hash);

        repo.expect_get_existing_hashes().returning(move |_| {
            let e = existing.clone();
            Box::pin(async move { Ok(e) })
        });

        repo.expect_upsert_vectors()
            .withf(|vectors, _, _| vectors.len() == 2)
            .returning(|vectors, _, _| {
                let len = vectors.len();
                Box::pin(async move { Ok(len) })
            });

        let service = IndexService::new(embedding, repo);
        let notes = vec![make_note(1, "hello"), make_note(2, "world")];
        let stats = service.index_notes(&notes, true).await.unwrap();

        assert_eq!(stats.notes_processed, 2);
        assert_eq!(stats.notes_embedded, 2);
        assert_eq!(stats.notes_skipped, 0);
    }

    // ====================================================================
    // IndexService::index_notes - payload correctness
    // ====================================================================

    #[tokio::test]
    async fn index_notes_builds_correct_payloads() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        repo.expect_get_existing_hashes()
            .returning(|_| Box::pin(async { Ok(HashMap::new()) }));

        repo.expect_upsert_vectors()
            .withf(|_, payloads, _| {
                let p = &payloads[0];
                p.note_id == 42
                    && p.model_id == 1
                    && p.deck_names == vec!["Default"]
                    && p.tags == vec!["tag1"]
                    && !p.content_hash.is_empty()
                    && !p.mature
                    && p.lapses == 0
                    && p.reps == 0
            })
            .returning(|vectors, _, _| {
                let len = vectors.len();
                Box::pin(async move { Ok(len) })
            });

        let service = IndexService::new(embedding, repo);
        let notes = vec![make_note(42, "test content")];
        service.index_notes(&notes, false).await.unwrap();
    }

    #[tokio::test]
    async fn index_notes_payload_includes_content_hash() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        repo.expect_get_existing_hashes()
            .returning(|_| Box::pin(async { Ok(HashMap::new()) }));

        let expected_hash = crate::embeddings::content_hash("mock/test", "my text");

        repo.expect_upsert_vectors()
            .withf(move |_, payloads, _| payloads[0].content_hash == expected_hash)
            .returning(|vectors, _, _| {
                let len = vectors.len();
                Box::pin(async move { Ok(len) })
            });

        let service = IndexService::new(embedding, repo);
        let notes = vec![make_note(1, "my text")];
        service.index_notes(&notes, false).await.unwrap();
    }

    // ====================================================================
    // IndexService::index_notes - embedding dimension correctness
    // ====================================================================

    #[tokio::test]
    async fn index_notes_vectors_have_correct_dimension() {
        let dim = 8;
        let embedding = MockEmbeddingProvider::new(dim);
        let mut repo = MockVectorRepository::new();

        repo.expect_get_existing_hashes()
            .returning(|_| Box::pin(async { Ok(HashMap::new()) }));

        repo.expect_upsert_vectors()
            .withf(move |vectors, _, _| vectors.iter().all(|v| v.len() == dim))
            .returning(|vectors, _, _| {
                let len = vectors.len();
                Box::pin(async move { Ok(len) })
            });

        let service = IndexService::new(embedding, repo);
        let notes = vec![make_note(1, "test")];
        service.index_notes(&notes, false).await.unwrap();
    }

    // ====================================================================
    // IndexService::index_notes - sparse vectors
    // ====================================================================

    #[tokio::test]
    async fn index_notes_generates_sparse_vectors() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        repo.expect_get_existing_hashes()
            .returning(|_| Box::pin(async { Ok(HashMap::new()) }));

        repo.expect_upsert_vectors()
            .withf(|_, _, sparse| sparse.is_some() && !sparse.unwrap().is_empty())
            .returning(|vectors, _, _| {
                let len = vectors.len();
                Box::pin(async move { Ok(len) })
            });

        let service = IndexService::new(embedding, repo);
        let notes = vec![make_note(1, "hello world")];
        service.index_notes(&notes, false).await.unwrap();
    }

    // ====================================================================
    // IndexService::index_notes - error propagation
    // ====================================================================

    #[tokio::test]
    async fn index_notes_propagates_embedding_error() {
        use async_trait::async_trait;

        struct FailingProvider;

        #[async_trait]
        impl EmbeddingProvider for FailingProvider {
            fn model_name(&self) -> &str {
                "fail/test"
            }
            fn dimension(&self) -> usize {
                4
            }
            async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
                Err(EmbeddingError::BatchFailed {
                    source: "test error".into(),
                })
            }
        }

        let mut repo = MockVectorRepository::new();
        repo.expect_get_existing_hashes()
            .returning(|_| Box::pin(async { Ok(HashMap::new()) }));

        let service = IndexService::new(FailingProvider, repo);
        let notes = vec![make_note(1, "test")];
        let result = service.index_notes(&notes, false).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), IndexError::Embedding(_)));
    }

    #[tokio::test]
    async fn index_notes_propagates_vector_store_error() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        repo.expect_get_existing_hashes()
            .returning(|_| Box::pin(async { Ok(HashMap::new()) }));

        repo.expect_upsert_vectors().returning(|_, _, _| {
            Box::pin(async { Err(VectorStoreError::Client("upsert failed".into())) })
        });

        let service = IndexService::new(embedding, repo);
        let notes = vec![make_note(1, "test")];
        let result = service.index_notes(&notes, false).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), IndexError::VectorStore(_)));
    }

    // ====================================================================
    // IndexService::delete_notes
    // ====================================================================

    #[tokio::test]
    async fn delete_notes_delegates_to_repo() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        repo.expect_delete_vectors()
            .withf(|ids| ids == &[10, 20, 30])
            .returning(|ids| {
                let len = ids.len();
                Box::pin(async move { Ok(len) })
            });

        let service = IndexService::new(embedding, repo);
        let count = service.delete_notes(&[10, 20, 30]).await.unwrap();
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn delete_notes_empty_input() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        repo.expect_delete_vectors()
            .withf(|ids| ids.is_empty())
            .returning(|_| Box::pin(async { Ok(0) }));

        let service = IndexService::new(embedding, repo);
        let count = service.delete_notes(&[]).await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn delete_notes_propagates_error() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        repo.expect_delete_vectors().returning(|_| {
            Box::pin(async { Err(VectorStoreError::Client("delete failed".into())) })
        });

        let service = IndexService::new(embedding, repo);
        let result = service.delete_notes(&[1]).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), IndexError::VectorStore(_)));
    }

    // ====================================================================
    // IndexService::index_notes - multiple notes with mixed hashes
    // ====================================================================

    #[tokio::test]
    async fn index_notes_mixed_new_and_existing() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        let hash1 = crate::embeddings::content_hash("mock/test", "text1");
        let hash3 = crate::embeddings::content_hash("mock/test", "text3");
        let mut existing = HashMap::new();
        existing.insert(1_i64, hash1);
        existing.insert(3_i64, hash3);

        repo.expect_get_existing_hashes().returning(move |_| {
            let e = existing.clone();
            Box::pin(async move { Ok(e) })
        });

        repo.expect_upsert_vectors()
            .withf(|vectors, payloads, _| {
                vectors.len() == 2 && payloads.iter().all(|p| p.note_id == 2 || p.note_id == 4)
            })
            .returning(|vectors, _, _| {
                let len = vectors.len();
                Box::pin(async move { Ok(len) })
            });

        let service = IndexService::new(embedding, repo);
        let notes = vec![
            make_note(1, "text1"),
            make_note(2, "text2"),
            make_note(3, "text3"),
            make_note(4, "text4"),
        ];
        let stats = service.index_notes(&notes, false).await.unwrap();

        assert_eq!(stats.notes_processed, 4);
        assert_eq!(stats.notes_embedded, 2);
        assert_eq!(stats.notes_skipped, 2);
    }

    // ====================================================================
    // IndexService::index_notes - note metadata propagation
    // ====================================================================

    #[tokio::test]
    async fn index_notes_preserves_note_metadata_in_payload() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        repo.expect_get_existing_hashes()
            .returning(|_| Box::pin(async { Ok(HashMap::new()) }));

        repo.expect_upsert_vectors()
            .withf(|_, payloads, _| {
                let p = &payloads[0];
                p.mature
                    && p.lapses == 5
                    && p.reps == 20
                    && p.fail_rate == Some(0.25)
                    && p.tags == vec!["study"]
                    && p.deck_names == vec!["Math"]
            })
            .returning(|vectors, _, _| {
                let len = vectors.len();
                Box::pin(async move { Ok(len) })
            });

        let service = IndexService::new(embedding, repo);
        let note = NoteForIndexing {
            note_id: 100,
            model_id: 2,
            normalized_text: "calculus integral".to_string(),
            tags: vec!["study".to_string()],
            deck_names: vec!["Math".to_string()],
            mature: true,
            lapses: 5,
            reps: 20,
            fail_rate: Some(0.25),
        };
        service.index_notes(&[note], false).await.unwrap();
    }

    // ====================================================================
    // IndexService::index_notes - hash changes detect content changes
    // ====================================================================

    #[tokio::test]
    async fn index_notes_reembeds_when_hash_differs() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        let mut existing = HashMap::new();
        existing.insert(1_i64, "stale_hash_value".to_string());

        repo.expect_get_existing_hashes().returning(move |_| {
            let e = existing.clone();
            Box::pin(async move { Ok(e) })
        });

        repo.expect_upsert_vectors()
            .withf(|vectors, payloads, _| vectors.len() == 1 && payloads[0].note_id == 1)
            .returning(|vectors, _, _| {
                let len = vectors.len();
                Box::pin(async move { Ok(len) })
            });

        let service = IndexService::new(embedding, repo);
        let notes = vec![make_note(1, "updated content")];
        let stats = service.index_notes(&notes, false).await.unwrap();

        assert_eq!(stats.notes_embedded, 1);
        assert_eq!(stats.notes_skipped, 0);
    }

    // ====================================================================
    // Full roundtrip: index + verify interaction
    // ====================================================================

    #[tokio::test]
    async fn full_roundtrip_index_and_delete() {
        let embedding = MockEmbeddingProvider::new(4);
        let mut repo = MockVectorRepository::new();

        repo.expect_get_existing_hashes()
            .returning(|_| Box::pin(async { Ok(HashMap::new()) }));
        repo.expect_upsert_vectors().returning(|vectors, _, _| {
            let len = vectors.len();
            Box::pin(async move { Ok(len) })
        });

        repo.expect_delete_vectors()
            .withf(|ids| ids == &[1, 2])
            .returning(|ids| {
                let len = ids.len();
                Box::pin(async move { Ok(len) })
            });

        let service = IndexService::new(embedding, repo);

        let notes = vec![make_note(1, "first"), make_note(2, "second")];
        let stats = service.index_notes(&notes, false).await.unwrap();
        assert_eq!(stats.notes_embedded, 2);

        let deleted = service.delete_notes(&[1, 2]).await.unwrap();
        assert_eq!(deleted, 2);
    }
}

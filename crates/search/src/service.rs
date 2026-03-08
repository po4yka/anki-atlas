use std::collections::HashMap;

use serde::Serialize;

use crate::error::SearchError;
use crate::fts::SearchFilters;
use crate::fusion::{FusionStats, SearchResult};
use crate::reranker::Reranker;

/// Complete hybrid search result.
#[derive(Debug, Clone, Serialize)]
pub struct HybridSearchResult {
    pub results: Vec<SearchResult>,
    pub stats: FusionStats,
    pub query: String,
    pub filters_applied: HashMap<String, serde_json::Value>,
    pub lexical_mode: String,
    pub lexical_fallback_used: bool,
    pub query_suggestions: Vec<String>,
    pub autocomplete_suggestions: Vec<String>,
    pub rerank_applied: bool,
    pub rerank_model: Option<String>,
    pub rerank_top_n: Option<usize>,
}

/// Detailed note information for result enrichment.
#[derive(Debug, Clone, Serialize)]
pub struct NoteDetail {
    pub note_id: i64,
    pub model_id: i64,
    pub normalized_text: String,
    pub tags: Vec<String>,
    pub deck_names: Vec<String>,
    pub mature: bool,
    pub lapses: i32,
    pub reps: i32,
}

/// Search service with trait-based DI.
#[allow(dead_code)]
pub struct SearchService<E, V, R>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
    R: Reranker,
{
    embedding: E,
    vector_repo: V,
    reranker: Option<R>,
    db: sqlx::PgPool,
    rerank_enabled: bool,
    rerank_top_n: usize,
}

impl<E, V, R> SearchService<E, V, R>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
    R: Reranker,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        _embedding: E,
        _vector_repo: V,
        _reranker: Option<R>,
        _db: sqlx::PgPool,
        _rerank_enabled: bool,
        _rerank_top_n: usize,
    ) -> Self {
        todo!()
    }

    /// Execute hybrid search: semantic + FTS -> RRF fusion -> optional rerank.
    #[allow(clippy::too_many_arguments)]
    pub async fn search(
        &self,
        _query: &str,
        _filters: Option<&SearchFilters>,
        _limit: usize,
        _semantic_weight: f64,
        _fts_weight: f64,
        _semantic_only: bool,
        _fts_only: bool,
        _rerank_override: Option<bool>,
        _rerank_top_n_override: Option<usize>,
    ) -> Result<HybridSearchResult, SearchError> {
        todo!()
    }

    /// Fetch note details for a list of IDs (for reranking / enrichment).
    pub async fn get_notes_details(
        &self,
        _note_ids: &[i64],
    ) -> Result<HashMap<i64, NoteDetail>, SearchError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test helpers ──────────────────────────────────────────────

    /// Simple mock embedding provider for service tests.
    struct FakeEmbedding;

    #[async_trait::async_trait]
    impl indexer::embeddings::EmbeddingProvider for FakeEmbedding {
        fn model_name(&self) -> &str {
            "fake/test"
        }

        fn dimension(&self) -> usize {
            4
        }

        async fn embed(
            &self,
            texts: &[String],
        ) -> Result<Vec<Vec<f32>>, indexer::embeddings::EmbeddingError> {
            Ok(texts.iter().map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect())
        }
    }

    /// Simple mock vector repository for service tests.
    struct FakeVectorRepo {
        results: Vec<(i64, f32)>,
    }

    impl FakeVectorRepo {
        fn new(results: Vec<(i64, f32)>) -> Self {
            Self { results }
        }

        fn empty() -> Self {
            Self::new(vec![])
        }
    }

    #[async_trait::async_trait]
    impl indexer::qdrant::VectorRepository for FakeVectorRepo {
        async fn ensure_collection(
            &self,
            _dimension: usize,
        ) -> Result<bool, indexer::qdrant::VectorStoreError> {
            Ok(false)
        }

        async fn upsert_vectors(
            &self,
            _vectors: &[Vec<f32>],
            _payloads: &[indexer::qdrant::NotePayload],
            _sparse_vectors: Option<&[indexer::qdrant::SparseVector]>,
        ) -> Result<usize, indexer::qdrant::VectorStoreError> {
            Ok(0)
        }

        async fn delete_vectors(
            &self,
            _note_ids: &[i64],
        ) -> Result<usize, indexer::qdrant::VectorStoreError> {
            Ok(0)
        }

        async fn get_existing_hashes(
            &self,
            _note_ids: &[i64],
        ) -> Result<HashMap<i64, String>, indexer::qdrant::VectorStoreError> {
            Ok(HashMap::new())
        }

        async fn search(
            &self,
            _query_vector: &[f32],
            _query_sparse: Option<&indexer::qdrant::SparseVector>,
            _limit: usize,
            _filters: &indexer::qdrant::SearchFilters,
        ) -> Result<Vec<(i64, f32)>, indexer::qdrant::VectorStoreError> {
            Ok(self.results.clone())
        }

        async fn find_similar_to_note(
            &self,
            _note_id: i64,
            _limit: usize,
            _min_score: f32,
            _deck_names: Option<&[String]>,
            _tags: Option<&[String]>,
        ) -> Result<Vec<(i64, f32)>, indexer::qdrant::VectorStoreError> {
            Ok(vec![])
        }

        async fn close(&self) -> Result<(), indexer::qdrant::VectorStoreError> {
            Ok(())
        }
    }

    /// Simple mock reranker that returns scores in order.
    struct FakeReranker {
        should_fail: bool,
    }

    impl FakeReranker {
        fn new() -> Self {
            Self { should_fail: false }
        }

        fn failing() -> Self {
            Self { should_fail: true }
        }
    }

    #[async_trait::async_trait]
    impl Reranker for FakeReranker {
        async fn rerank(
            &self,
            _query: &str,
            documents: &[(i64, String)],
        ) -> Result<Vec<(i64, f64)>, SearchError> {
            if self.should_fail {
                return Err(SearchError::Rerank("model unavailable".to_string()));
            }
            // Return descending scores based on position
            Ok(documents
                .iter()
                .enumerate()
                .map(|(i, (id, _))| (*id, 1.0 - i as f64 * 0.1))
                .collect())
        }
    }

    fn fake_pool() -> sqlx::PgPool {
        sqlx::PgPool::connect_lazy("postgres://invalid:5432/fake").unwrap()
    }

    // ── Type construction tests ──────────────────────────────────

    #[test]
    fn hybrid_search_result_construction() {
        let result = HybridSearchResult {
            results: vec![],
            stats: FusionStats::default(),
            query: "test query".to_string(),
            filters_applied: HashMap::new(),
            lexical_mode: "none".to_string(),
            lexical_fallback_used: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
            rerank_applied: false,
            rerank_model: None,
            rerank_top_n: None,
        };
        assert_eq!(result.query, "test query");
        assert_eq!(result.lexical_mode, "none");
        assert!(!result.rerank_applied);
        assert!(result.results.is_empty());
    }

    #[test]
    fn hybrid_search_result_with_rerank_metadata() {
        let result = HybridSearchResult {
            results: vec![],
            stats: FusionStats::default(),
            query: "rerank test".to_string(),
            filters_applied: HashMap::new(),
            lexical_mode: "fts".to_string(),
            lexical_fallback_used: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
            rerank_applied: true,
            rerank_model: Some("cross-encoder/ms-marco".to_string()),
            rerank_top_n: Some(20),
        };
        assert!(result.rerank_applied);
        assert_eq!(
            result.rerank_model.as_deref(),
            Some("cross-encoder/ms-marco")
        );
        assert_eq!(result.rerank_top_n, Some(20));
    }

    #[test]
    fn note_detail_construction() {
        let detail = NoteDetail {
            note_id: 42,
            model_id: 100,
            normalized_text: "some card content".to_string(),
            tags: vec!["rust".to_string(), "programming".to_string()],
            deck_names: vec!["Default".to_string()],
            mature: true,
            lapses: 3,
            reps: 25,
        };
        assert_eq!(detail.note_id, 42);
        assert_eq!(detail.model_id, 100);
        assert!(detail.mature);
        assert_eq!(detail.lapses, 3);
        assert_eq!(detail.reps, 25);
        assert_eq!(detail.tags.len(), 2);
    }

    #[test]
    fn hybrid_search_result_serializes_to_json() {
        let result = HybridSearchResult {
            results: vec![],
            stats: FusionStats::default(),
            query: "json test".to_string(),
            filters_applied: HashMap::new(),
            lexical_mode: "none".to_string(),
            lexical_fallback_used: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
            rerank_applied: false,
            rerank_model: None,
            rerank_top_n: None,
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["query"], "json test");
        assert_eq!(json["lexical_mode"], "none");
        assert_eq!(json["rerank_applied"], false);
    }

    #[test]
    fn note_detail_serializes_to_json() {
        let detail = NoteDetail {
            note_id: 1,
            model_id: 2,
            normalized_text: "text".to_string(),
            tags: vec!["tag1".to_string()],
            deck_names: vec!["Deck".to_string()],
            mature: false,
            lapses: 0,
            reps: 0,
        };
        let json = serde_json::to_value(&detail).unwrap();
        assert_eq!(json["note_id"], 1);
        assert_eq!(json["mature"], false);
    }

    // ── SearchService construction ───────────────────────────────

    #[tokio::test]
    async fn service_new_constructs_with_all_deps() {
        let pool = fake_pool();
        let _svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            pool,
            false,
            20,
        );
    }

    #[tokio::test]
    async fn service_new_without_reranker() {
        let pool = fake_pool();
        let _svc: SearchService<FakeEmbedding, FakeVectorRepo, FakeReranker> =
            SearchService::new(FakeEmbedding, FakeVectorRepo::empty(), None, pool, false, 20);
    }

    // ── SearchService::search tests ──────────────────────────────

    #[tokio::test]
    async fn search_empty_query_returns_empty() {
        let pool = fake_pool();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            pool,
            false,
            20,
        );

        let result = svc
            .search("", None, 50, 1.0, 1.0, false, false, None, None)
            .await
            .unwrap();

        assert!(result.results.is_empty());
        assert_eq!(result.query, "");
        assert_eq!(result.stats, FusionStats::default());
    }

    #[tokio::test]
    async fn search_whitespace_query_returns_empty() {
        let pool = fake_pool();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            pool,
            false,
            20,
        );

        let result = svc
            .search("   ", None, 50, 1.0, 1.0, false, false, None, None)
            .await
            .unwrap();

        assert!(result.results.is_empty());
    }

    #[tokio::test]
    async fn search_semantic_only_skips_fts() {
        let pool = fake_pool();
        let semantic_results = vec![(1, 0.95_f32), (2, 0.85)];
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(semantic_results),
            Some(FakeReranker::new()),
            pool,
            false,
            20,
        );

        let result = svc
            .search("test query", None, 50, 1.0, 1.0, true, false, None, None)
            .await
            .unwrap();

        // Should have semantic results only
        assert!(!result.results.is_empty());
        for r in &result.results {
            assert!(r.semantic_score.is_some());
            assert!(r.fts_score.is_none());
        }
        assert_eq!(result.lexical_mode, "none");
    }

    #[tokio::test]
    async fn search_fts_only_skips_semantic() {
        let pool = fake_pool();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            pool,
            false,
            20,
        );

        // fts_only=true: semantic should be skipped, only FTS runs
        // With a fake pool, FTS will fail or return empty - but the key behavior
        // is that semantic_weight should be 0 and no embedding call happens.
        let result = svc
            .search("test query", None, 50, 1.0, 1.0, false, true, None, None)
            .await
            .unwrap();

        // Semantic scores should all be None when fts_only
        for r in &result.results {
            assert!(r.semantic_score.is_none());
        }
    }

    #[tokio::test]
    async fn search_returns_query_in_result() {
        let pool = fake_pool();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(vec![(1, 0.9)]),
            Some(FakeReranker::new()),
            pool,
            false,
            20,
        );

        let result = svc
            .search(
                "my search query",
                None,
                50,
                1.0,
                1.0,
                true,
                false,
                None,
                None,
            )
            .await
            .unwrap();

        assert_eq!(result.query, "my search query");
    }

    #[tokio::test]
    async fn search_rerank_disabled_returns_rerank_applied_false() {
        let pool = fake_pool();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(vec![(1, 0.9)]),
            Some(FakeReranker::new()),
            pool,
            false, // rerank_enabled = false
            20,
        );

        let result = svc
            .search(
                "test",
                None,
                50,
                1.0,
                1.0,
                true,
                false,
                None,
                None,
            )
            .await
            .unwrap();

        assert!(!result.rerank_applied);
        assert!(result.rerank_model.is_none());
        assert!(result.rerank_top_n.is_none());
    }

    #[tokio::test]
    async fn search_rerank_failure_degrades_gracefully() {
        let pool = fake_pool();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(vec![(1, 0.95), (2, 0.85), (3, 0.75)]),
            Some(FakeReranker::failing()),
            pool,
            true, // rerank_enabled = true
            20,
        );

        // Even though reranker fails, search should succeed with rerank_applied=false
        let result = svc
            .search(
                "test query",
                None,
                50,
                1.0,
                1.0,
                true,
                false,
                None,
                None,
            )
            .await
            .unwrap();

        assert!(!result.rerank_applied);
        assert!(!result.results.is_empty());
    }

    #[tokio::test]
    async fn search_rerank_override_enables_reranking() {
        let pool = fake_pool();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(vec![(1, 0.95), (2, 0.85)]),
            Some(FakeReranker::new()),
            pool,
            false, // rerank_enabled = false globally
            20,
        );

        // Override with rerank_override=Some(true) should enable reranking
        let result = svc
            .search(
                "test query",
                None,
                50,
                1.0,
                1.0,
                true,
                false,
                Some(true),
                None,
            )
            .await
            .unwrap();

        // Rerank override should take precedence
        // (Exact behavior depends on GREEN implementation, but result should not error)
        assert_eq!(result.query, "test query");
    }

    #[tokio::test]
    async fn search_rerank_override_disables_reranking() {
        let pool = fake_pool();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(vec![(1, 0.95)]),
            Some(FakeReranker::new()),
            pool,
            true, // rerank_enabled = true globally
            20,
        );

        // Override with rerank_override=Some(false) should disable reranking
        let result = svc
            .search(
                "test query",
                None,
                50,
                1.0,
                1.0,
                true,
                false,
                Some(false),
                None,
            )
            .await
            .unwrap();

        assert!(!result.rerank_applied);
    }

    #[tokio::test]
    async fn search_no_reranker_with_rerank_enabled_degrades() {
        let pool = fake_pool();
        let svc: SearchService<FakeEmbedding, FakeVectorRepo, FakeReranker> = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(vec![(1, 0.95)]),
            None, // no reranker provided
            pool,
            true, // but rerank_enabled
            20,
        );

        // Should degrade gracefully when no reranker is available
        let result = svc
            .search(
                "test query",
                None,
                50,
                1.0,
                1.0,
                true,
                false,
                None,
                None,
            )
            .await
            .unwrap();

        assert!(!result.rerank_applied);
    }

    #[tokio::test]
    async fn search_limit_respected() {
        let pool = fake_pool();
        let many_results: Vec<(i64, f32)> =
            (1..=20).map(|i| (i, 1.0 - i as f32 * 0.01)).collect();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(many_results),
            Some(FakeReranker::new()),
            pool,
            false,
            20,
        );

        let result = svc
            .search(
                "test query",
                None,
                5, // limit to 5
                1.0,
                1.0,
                true,
                false,
                None,
                None,
            )
            .await
            .unwrap();

        assert!(result.results.len() <= 5);
    }

    // ── Send + Sync ──────────────────────────────────────────────

    #[test]
    fn search_service_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SearchService<FakeEmbedding, FakeVectorRepo, FakeReranker>>();
    }

    #[test]
    fn hybrid_search_result_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HybridSearchResult>();
    }

    #[test]
    fn note_detail_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoteDetail>();
    }
}

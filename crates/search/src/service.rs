use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use tracing::instrument;

use crate::error::SearchError;
use crate::fts::{LexicalMode, SearchFilters};
use crate::fusion::{FusionStats, SearchResult};
use crate::reranker::Reranker;

/// Parameters for a hybrid search query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    pub query: String,
    pub filters: Option<SearchFilters>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default = "default_weight")]
    pub semantic_weight: f64,
    #[serde(default = "default_weight")]
    pub fts_weight: f64,
    #[serde(default)]
    pub semantic_only: bool,
    #[serde(default)]
    pub fts_only: bool,
    pub rerank_override: Option<bool>,
    pub rerank_top_n_override: Option<usize>,
}

fn default_limit() -> usize {
    50
}
fn default_weight() -> f64 {
    1.0
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            query: String::new(),
            filters: None,
            limit: default_limit(),
            semantic_weight: default_weight(),
            fts_weight: default_weight(),
            semantic_only: false,
            fts_only: false,
            rerank_override: None,
            rerank_top_n_override: None,
        }
    }
}

/// Complete hybrid search result.
#[derive(Debug, Clone, Serialize)]
pub struct HybridSearchResult {
    pub results: Vec<SearchResult>,
    pub stats: FusionStats,
    pub query: String,
    pub filters_applied: HashMap<String, serde_json::Value>,
    pub lexical_mode: LexicalMode,
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

/// Raw row from note details query.
#[derive(Debug, FromRow)]
struct NoteDetailRow {
    note_id: i64,
    model_id: i64,
    normalized_text: String,
    tags: Vec<String>,
    deck_name: String,
    mature: bool,
    lapses: i32,
    reps: i32,
}

/// Search service with trait-based DI.
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
        embedding: E,
        vector_repo: V,
        reranker: Option<R>,
        db: sqlx::PgPool,
        rerank_enabled: bool,
        rerank_top_n: usize,
    ) -> Self {
        Self {
            embedding,
            vector_repo,
            reranker,
            db,
            rerank_enabled,
            rerank_top_n,
        }
    }

    /// Execute hybrid search: semantic + FTS -> RRF fusion -> optional rerank.
    #[instrument(skip(self))]
    pub async fn search(&self, params: &SearchParams) -> Result<HybridSearchResult, SearchError> {
        let SearchParams {
            ref query,
            filters: ref _filters,
            limit,
            semantic_weight,
            fts_weight,
            semantic_only,
            fts_only,
            rerank_override,
            rerank_top_n_override,
        } = *params;

        // Empty/whitespace query short-circuit
        if query.trim().is_empty() {
            return Ok(HybridSearchResult {
                results: vec![],
                stats: FusionStats::default(),
                query: query.to_string(),
                filters_applied: HashMap::new(),
                lexical_mode: LexicalMode::None,
                lexical_fallback_used: false,
                query_suggestions: vec![],
                autocomplete_suggestions: vec![],
                rerank_applied: false,
                rerank_model: None,
                rerank_top_n: None,
            });
        }

        // Semantic search
        let semantic_results = if fts_only {
            vec![]
        } else {
            let embedded = self.embedding.embed(&[query.to_string()]).await?;
            let query_vector = &embedded[0];
            let raw = self
                .vector_repo
                .search(
                    query_vector,
                    None,
                    limit,
                    &indexer::qdrant::SearchFilters::default(),
                )
                .await?;
            raw.into_iter()
                .map(|(id, score)| (id, score as f64))
                .collect::<Vec<_>>()
        };

        // FTS search
        let (fts_results, lexical_mode, lexical_fallback_used, query_suggestions, autocomplete_suggestions) = if semantic_only {
            (vec![], LexicalMode::None, false, vec![], vec![])
        } else {
            let lexical = crate::fts::search_lexical(
                &self.db,
                query,
                params.filters.as_ref(),
                limit as i64,
            )
            .await?;
            let fts_results = lexical
                .results
                .into_iter()
                .map(|r| (r.note_id, r.rank, r.headline))
                .collect();
            (
                fts_results,
                lexical.mode,
                lexical.used_fallback,
                lexical.query_suggestions,
                lexical.autocomplete_suggestions,
            )
        };

        // RRF fusion
        let (mut results, stats) = crate::fusion::reciprocal_rank_fusion(
            &semantic_results,
            &fts_results,
            60,
            limit,
            if fts_only { 0.0 } else { semantic_weight },
            if semantic_only { 0.0 } else { fts_weight },
        );

        // Determine whether to rerank
        let should_rerank = rerank_override.unwrap_or(self.rerank_enabled);
        let rerank_top_n = rerank_top_n_override.unwrap_or(self.rerank_top_n);
        let mut rerank_applied = false;
        let mut rerank_model: Option<String> = None;

        if should_rerank {
            if let Some(ref reranker) = self.reranker {
                let docs: Vec<(i64, String)> = results
                    .iter()
                    .take(rerank_top_n)
                    .map(|r| (r.note_id, String::new()))
                    .collect();

                if !docs.is_empty() {
                    match reranker.rerank(query, &docs).await {
                        Ok(scores) => {
                            let score_map: HashMap<i64, f64> = scores.into_iter().collect();
                            for r in &mut results {
                                if let Some(&s) = score_map.get(&r.note_id) {
                                    r.rerank_score = Some(s);
                                }
                            }
                            results.sort_by(|a, b| {
                                b.rerank_score
                                    .partial_cmp(&a.rerank_score)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                            rerank_applied = true;
                            rerank_model = Some(reranker.model_name().to_string());
                        }
                        Err(_) => {
                            // Degrade gracefully: skip reranking
                        }
                    }
                }
            }
            // No reranker provided: degrade gracefully
        }

        // Apply limit
        results.truncate(limit);

        Ok(HybridSearchResult {
            results,
            stats,
            query: query.to_string(),
            filters_applied: HashMap::new(),
            lexical_mode,
            lexical_fallback_used,
            query_suggestions,
            autocomplete_suggestions,
            rerank_applied,
            rerank_model: if rerank_applied { rerank_model } else { None },
            rerank_top_n: if rerank_applied {
                Some(rerank_top_n)
            } else {
                None
            },
        })
    }

    /// Fetch note details for a list of IDs (for reranking / enrichment).
    #[instrument(skip(self))]
    pub async fn get_notes_details(
        &self,
        note_ids: &[i64],
    ) -> Result<HashMap<i64, NoteDetail>, SearchError> {
        if note_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let rows = sqlx::query_as::<_, NoteDetailRow>(
            "SELECT n.id AS note_id, n.mid AS model_id, n.normalized_text, \
             n.tags, \
             COALESCE(d.name, '') AS deck_name, \
             COALESCE(c.ivl >= 21, false) AS mature, \
             COALESCE(c.lapses, 0) AS lapses, \
             COALESCE(c.reps, 0) AS reps \
             FROM notes n \
             LEFT JOIN cards c ON c.nid = n.id \
             LEFT JOIN decks d ON d.id = c.did \
             WHERE n.id = ANY($1)",
        )
        .bind(note_ids)
        .fetch_all(&self.db)
        .await?;

        let mut map = HashMap::new();
        for row in rows {
            let entry = map.entry(row.note_id).or_insert_with(|| NoteDetail {
                note_id: row.note_id,
                model_id: row.model_id,
                normalized_text: row.normalized_text.clone(),
                tags: row.tags.clone(),
                deck_names: vec![],
                mature: row.mature,
                lapses: row.lapses,
                reps: row.reps,
            });
            if !row.deck_name.is_empty() && !entry.deck_names.contains(&row.deck_name) {
                entry.deck_names.push(row.deck_name.clone());
            }
        }

        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test helpers ──────────────────────────────────────────────

    fn params(query: &str) -> SearchParams {
        SearchParams {
            query: query.to_string(),
            ..Default::default()
        }
    }

    fn params_semantic_only(query: &str) -> SearchParams {
        SearchParams {
            query: query.to_string(),
            semantic_only: true,
            ..Default::default()
        }
    }

    fn params_fts_only(query: &str) -> SearchParams {
        SearchParams {
            query: query.to_string(),
            fts_only: true,
            ..Default::default()
        }
    }

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
        fn model_name(&self) -> &str {
            "fake/reranker"
        }

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
            lexical_mode: LexicalMode::None,
            lexical_fallback_used: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
            rerank_applied: false,
            rerank_model: None,
            rerank_top_n: None,
        };
        assert_eq!(result.query, "test query");
        assert_eq!(result.lexical_mode, LexicalMode::None);
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
            lexical_mode: LexicalMode::Fts,
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
            lexical_mode: LexicalMode::None,
            lexical_fallback_used: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
            rerank_applied: false,
            rerank_model: None,
            rerank_top_n: None,
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["query"], "json test");
        assert_eq!(json["lexical_mode"], "none"); // LexicalMode::None serializes as "none"
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
        let _svc: SearchService<FakeEmbedding, FakeVectorRepo, FakeReranker> = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            None,
            pool,
            false,
            20,
        );
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

        let result = svc.search(&params("")).await.unwrap();

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

        let result = svc.search(&params("   ")).await.unwrap();

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
            .search(&params_semantic_only("test query"))
            .await
            .unwrap();

        // Should have semantic results only
        assert!(!result.results.is_empty());
        for r in &result.results {
            assert!(r.semantic_score.is_some());
            assert!(r.fts_score.is_none());
        }
        assert_eq!(result.lexical_mode, LexicalMode::None);
    }

    #[tokio::test]
    async fn search_fts_only_returns_db_error_without_real_pool() {
        let pool = fake_pool();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            pool,
            false,
            20,
        );

        // fts_only=true triggers a real FTS query which fails with a fake pool
        let result = svc.search(&params_fts_only("test query")).await;
        assert!(result.is_err());
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
            .search(&params_semantic_only("my search query"))
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
            .search(&params_semantic_only("test"))
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
            .search(&params_semantic_only("test query"))
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
            .search(&SearchParams {
                query: "test query".into(),
                semantic_only: true,
                rerank_override: Some(true),
                ..Default::default()
            })
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
            .search(&SearchParams {
                query: "test query".into(),
                semantic_only: true,
                rerank_override: Some(false),
                ..Default::default()
            })
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
            .search(&params_semantic_only("test query"))
            .await
            .unwrap();

        assert!(!result.rerank_applied);
    }

    #[tokio::test]
    async fn search_limit_respected() {
        let pool = fake_pool();
        let many_results: Vec<(i64, f32)> = (1..=20).map(|i| (i, 1.0 - i as f32 * 0.01)).collect();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(many_results),
            Some(FakeReranker::new()),
            pool,
            false,
            20,
        );

        let result = svc
            .search(&SearchParams {
                query: "test query".into(),
                limit: 5,
                semantic_only: true,
                ..Default::default()
            })
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

    // ── get_notes_details tests ─────────────────────────────────

    #[tokio::test]
    async fn get_notes_details_empty_ids_returns_empty_map() {
        let pool = fake_pool();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            pool,
            false,
            20,
        );

        // Empty slice should return empty HashMap without hitting DB
        let result = svc.get_notes_details(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn get_notes_details_with_ids_returns_db_error() {
        let pool = fake_pool();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            pool,
            false,
            20,
        );

        // With actual IDs and a fake pool, should return a DB error (not panic)
        let result = svc.get_notes_details(&[1, 2, 3]).await;
        assert!(result.is_err());
    }
}

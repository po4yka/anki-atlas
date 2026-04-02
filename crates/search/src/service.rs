use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::error::SearchError;
use crate::fts::{LexicalMode, SearchFilters};
use crate::fusion::{FusionStats, SearchResult};
use crate::repository::SearchReadRepository;
use crate::reranker::Reranker;

/// Controls which retrieval sources are used during search.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    #[default]
    Hybrid,
    SemanticOnly,
    FtsOnly,
}

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
    pub search_mode: SearchMode,
    pub rerank_override: Option<bool>,
    pub rerank_top_n_override: Option<usize>,
}

/// Parameters for semantic chunk search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkSearchParams {
    pub query: String,
    pub filters: Option<SearchFilters>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

impl Default for ChunkSearchParams {
    fn default() -> Self {
        Self {
            query: String::new(),
            filters: None,
            limit: default_limit(),
        }
    }
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
            search_mode: SearchMode::Hybrid,
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

impl HybridSearchResult {
    /// Create an empty result for short-circuit returns (e.g. blank queries).
    pub fn empty(query: String) -> Self {
        Self {
            results: vec![],
            stats: FusionStats::default(),
            query,
            filters_applied: HashMap::new(),
            lexical_mode: LexicalMode::None,
            lexical_fallback_used: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
            rerank_applied: false,
            rerank_model: None,
            rerank_top_n: None,
        }
    }
}

/// Single semantic chunk hit.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkSearchHit {
    pub note_id: i64,
    pub chunk_id: String,
    pub chunk_kind: String,
    pub modality: String,
    pub source_field: Option<String>,
    pub asset_rel_path: Option<String>,
    pub mime_type: Option<String>,
    pub preview_label: Option<String>,
    pub score: f64,
}

/// Semantic-only chunk search result.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkSearchResult {
    pub query: String,
    pub results: Vec<ChunkSearchHit>,
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
pub struct SearchService<E, V, R>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
    R: Reranker,
{
    embedding: E,
    vector_repo: V,
    reranker: Option<R>,
    repository: Arc<dyn SearchReadRepository>,
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
        repository: Arc<dyn SearchReadRepository>,
        rerank_enabled: bool,
        rerank_top_n: usize,
    ) -> Self {
        Self {
            embedding,
            vector_repo,
            reranker,
            repository,
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
            search_mode,
            rerank_override,
            rerank_top_n_override,
        } = *params;

        // Empty/whitespace query short-circuit
        if query.trim().is_empty() {
            return Ok(HybridSearchResult::empty(query.to_string()));
        }

        // Semantic search
        let mut semantic_matches = HashMap::<i64, indexer::qdrant::SemanticSearchHit>::new();
        let semantic_results = if search_mode == SearchMode::FtsOnly {
            vec![]
        } else {
            let raw = crate::semantic::run_semantic_chunk_search(
                &self.embedding,
                &self.vector_repo,
                query,
                params.filters.as_ref(),
                limit,
            )
            .await?;
            let mut best_by_note = HashMap::<i64, f64>::new();
            for hit in raw {
                let score = f64::from(hit.score);
                best_by_note
                    .entry(hit.note_id)
                    .and_modify(|best_score| {
                        if score > *best_score {
                            *best_score = score;
                        }
                    })
                    .or_insert(score);
                semantic_matches
                    .entry(hit.note_id)
                    .and_modify(|existing| {
                        if hit.score > existing.score {
                            *existing = hit.clone();
                        }
                    })
                    .or_insert(hit);
            }
            let mut semantic_results: Vec<_> = best_by_note.into_iter().collect();
            semantic_results
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            semantic_results.truncate(limit);
            semantic_results
        };

        // FTS search
        let (
            fts_results,
            lexical_mode,
            lexical_fallback_used,
            query_suggestions,
            autocomplete_suggestions,
        ) = if search_mode == SearchMode::SemanticOnly {
            (vec![], LexicalMode::None, false, vec![], vec![])
        } else {
            let lexical = self
                .repository
                .search_lexical(query, params.filters.as_ref(), limit as i64)
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
            if search_mode == SearchMode::FtsOnly {
                0.0
            } else {
                semantic_weight
            },
            if search_mode == SearchMode::SemanticOnly {
                0.0
            } else {
                fts_weight
            },
        );

        for result in &mut results {
            if let Some(hit) = semantic_matches.get(&result.note_id) {
                result.match_modality = Some(hit.modality.clone());
                result.match_chunk_kind = Some(hit.chunk_kind.clone());
                result.match_source_field = hit.source_field.clone();
                result.match_asset_rel_path = hit.asset_rel_path.clone();
                result.match_preview_label = hit.preview_label.clone();
            }
        }

        // Determine whether to rerank
        let should_rerank = rerank_override.unwrap_or(self.rerank_enabled);
        let rerank_top_n = rerank_top_n_override.unwrap_or(self.rerank_top_n);
        let mut rerank_applied = false;
        let mut rerank_model: Option<String> = None;

        if should_rerank {
            if let Some(ref reranker) = self.reranker {
                (rerank_applied, rerank_model) = crate::reranking::apply_reranking(
                    &mut results,
                    query,
                    reranker,
                    rerank_top_n,
                    &self.repository,
                )
                .await;
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

    /// Execute semantic-only chunk search.
    #[instrument(skip(self))]
    pub async fn search_chunks(
        &self,
        params: &ChunkSearchParams,
    ) -> Result<ChunkSearchResult, SearchError> {
        if params.query.trim().is_empty() {
            return Ok(ChunkSearchResult {
                query: params.query.clone(),
                results: Vec::new(),
            });
        }

        let raw = crate::semantic::run_semantic_chunk_search(
            &self.embedding,
            &self.vector_repo,
            &params.query,
            params.filters.as_ref(),
            params.limit,
        )
        .await?;
        let mut results: Vec<_> = raw
            .into_iter()
            .map(|hit| ChunkSearchHit {
                note_id: hit.note_id,
                chunk_id: hit.chunk_id,
                chunk_kind: hit.chunk_kind,
                modality: hit.modality,
                source_field: hit.source_field,
                asset_rel_path: hit.asset_rel_path,
                mime_type: hit.mime_type,
                preview_label: hit.preview_label,
                score: f64::from(hit.score),
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(params.limit);

        Ok(ChunkSearchResult {
            query: params.query.clone(),
            results,
        })
    }

    /// Fetch note details for a list of IDs (for reranking / enrichment).
    #[instrument(skip(self))]
    pub async fn get_notes_details(
        &self,
        note_ids: &[i64],
    ) -> Result<HashMap<i64, NoteDetail>, SearchError> {
        self.repository.get_note_details(note_ids).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use indexer::embeddings::EmbeddingInput;

    use super::*;
    use crate::fts::{FtsResult, FtsSource, LexicalSearchResult};
    use crate::repository::{SearchReadRepository, SqlxSearchReadRepository};
    use database::run_migrations;
    use sqlx::postgres::PgPoolOptions;
    use testcontainers::runners::AsyncRunner;
    use testcontainers_modules::postgres::Postgres;

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
            search_mode: SearchMode::SemanticOnly,
            ..Default::default()
        }
    }

    fn params_fts_only(query: &str) -> SearchParams {
        SearchParams {
            query: query.to_string(),
            search_mode: SearchMode::FtsOnly,
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

        async fn embed_inputs(
            &self,
            inputs: &[EmbeddingInput],
        ) -> Result<Vec<Vec<f32>>, indexer::embeddings::EmbeddingError> {
            Ok(inputs.iter().map(|_| vec![0.1, 0.2, 0.3, 0.4]).collect())
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
        results: Vec<indexer::qdrant::SemanticSearchHit>,
    }

    impl FakeVectorRepo {
        fn new(results: Vec<(i64, f32)>) -> Self {
            Self {
                results: results
                    .into_iter()
                    .map(|(note_id, score)| indexer::qdrant::SemanticSearchHit {
                        note_id,
                        chunk_id: format!("{note_id}:text_primary"),
                        chunk_kind: "text_primary".to_string(),
                        modality: "text".to_string(),
                        source_field: None,
                        asset_rel_path: None,
                        mime_type: Some("text/plain".to_string()),
                        preview_label: Some(format!("note {note_id}")),
                        score,
                    })
                    .collect(),
            }
        }

        fn empty() -> Self {
            Self::new(vec![])
        }
    }

    struct FakeSearchReadRepository {
        lexical: LexicalSearchResult,
        lexical_error: bool,
        note_details: HashMap<i64, NoteDetail>,
        note_details_error: bool,
    }

    impl FakeSearchReadRepository {
        fn with_note_details(note_ids: &[i64]) -> Self {
            let note_details = note_ids
                .iter()
                .map(|note_id| {
                    (
                        *note_id,
                        NoteDetail {
                            note_id: *note_id,
                            model_id: 100,
                            normalized_text: format!("note {note_id} body"),
                            tags: vec!["rust".to_string()],
                            deck_names: vec!["Default".to_string()],
                            mature: true,
                            lapses: 0,
                            reps: 10,
                        },
                    )
                })
                .collect();
            Self {
                note_details,
                ..Default::default()
            }
        }

        fn with_lexical(results: Vec<FtsResult>, mode: LexicalMode) -> Self {
            Self {
                lexical: LexicalSearchResult {
                    results,
                    mode,
                    used_fallback: false,
                    query_suggestions: vec![],
                    autocomplete_suggestions: vec![],
                },
                ..Default::default()
            }
        }

        fn failing_lexical() -> Self {
            Self {
                lexical_error: true,
                ..Default::default()
            }
        }

        fn failing_note_details() -> Self {
            Self {
                note_details_error: true,
                ..Default::default()
            }
        }
    }

    impl Default for FakeSearchReadRepository {
        fn default() -> Self {
            Self {
                lexical: LexicalSearchResult {
                    results: vec![],
                    mode: LexicalMode::None,
                    used_fallback: false,
                    query_suggestions: vec![],
                    autocomplete_suggestions: vec![],
                },
                lexical_error: false,
                note_details: HashMap::new(),
                note_details_error: false,
            }
        }
    }

    #[async_trait::async_trait]
    impl SearchReadRepository for FakeSearchReadRepository {
        async fn search_lexical(
            &self,
            _query: &str,
            _filters: Option<&SearchFilters>,
            _limit: i64,
        ) -> Result<LexicalSearchResult, SearchError> {
            if self.lexical_error {
                return Err(SearchError::InvalidRequest(
                    "lexical lookup failed".to_string(),
                ));
            }
            Ok(self.lexical.clone())
        }

        async fn get_note_details(
            &self,
            note_ids: &[i64],
        ) -> Result<HashMap<i64, NoteDetail>, SearchError> {
            if self.note_details_error {
                return Err(SearchError::InvalidRequest(
                    "note detail lookup failed".to_string(),
                ));
            }
            Ok(note_ids
                .iter()
                .filter_map(|note_id| {
                    self.note_details
                        .get(note_id)
                        .cloned()
                        .map(|detail| (*note_id, detail))
                })
                .collect())
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

        async fn collection_dimension(
            &self,
        ) -> Result<Option<usize>, indexer::qdrant::VectorStoreError> {
            Ok(Some(4))
        }

        async fn recreate_collection(
            &self,
            _dimension: usize,
        ) -> Result<(), indexer::qdrant::VectorStoreError> {
            Ok(())
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

        async fn search_chunks(
            &self,
            _query_vector: &[f32],
            _query_sparse: Option<&indexer::qdrant::SparseVector>,
            _limit: usize,
            _filters: &indexer::qdrant::SearchFilters,
        ) -> Result<Vec<indexer::qdrant::SemanticSearchHit>, indexer::qdrant::VectorStoreError>
        {
            Ok(self.results.clone())
        }

        async fn find_similar_to_note(
            &self,
            _note_id: i64,
            _limit: usize,
            _min_score: f32,
            _deck_names: Option<&[String]>,
            _tags: Option<&[String]>,
        ) -> Result<Vec<indexer::qdrant::ScoredNote>, indexer::qdrant::VectorStoreError> {
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
                return Err(SearchError::from(crate::error::RerankError::Protocol {
                    message: "model unavailable".to_string(),
                }));
            }
            // Return descending scores based on position
            Ok(documents
                .iter()
                .enumerate()
                .map(|(i, (id, _))| (*id, 1.0 - i as f64 * 0.1))
                .collect())
        }
    }

    fn fake_repo() -> Arc<dyn SearchReadRepository> {
        Arc::new(FakeSearchReadRepository::default())
    }

    struct RecordingReranker {
        seen_documents: Arc<Mutex<Vec<(i64, String)>>>,
    }

    impl RecordingReranker {
        fn new(seen_documents: Arc<Mutex<Vec<(i64, String)>>>) -> Self {
            Self { seen_documents }
        }
    }

    #[async_trait::async_trait]
    impl Reranker for RecordingReranker {
        fn model_name(&self) -> &str {
            "recording/reranker"
        }

        async fn rerank(
            &self,
            _query: &str,
            documents: &[(i64, String)],
        ) -> Result<Vec<(i64, f64)>, SearchError> {
            *self.seen_documents.lock().unwrap() = documents.to_vec();
            Ok(documents
                .iter()
                .enumerate()
                .map(|(index, (note_id, _))| (*note_id, 1.0 - index as f64 * 0.1))
                .collect())
        }
    }

    async fn setup_real_pool() -> Option<(sqlx::PgPool, testcontainers::ContainerAsync<Postgres>)> {
        let container = Postgres::default().start().await.ok()?;
        let host = container.get_host().await.ok()?;
        let port = container.get_host_port_ipv4(5432).await.ok()?;
        let url = format!("postgresql://postgres:postgres@{host}:{port}/postgres");

        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&url)
            .await
            .ok()?;

        run_migrations(&pool).await.ok()?;
        Some((pool, container))
    }

    async fn seed_search_fixture(pool: &sqlx::PgPool) {
        sqlx::query("INSERT INTO decks (deck_id, name) VALUES ($1, $2), ($3, $4)")
            .bind(10_i64)
            .bind("Default")
            .bind(20_i64)
            .bind("Archive")
            .execute(pool)
            .await
            .unwrap();

        sqlx::query(
            "INSERT INTO notes \
             (note_id, model_id, tags, fields_json, raw_fields, normalized_text, mtime, usn) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8), \
                    ($9, $10, $11, $12, $13, $14, $15, $16)",
        )
        .bind(1_i64)
        .bind(100_i64)
        .bind(vec!["rust".to_string()])
        .bind(serde_json::json!({"Front": "What is ownership?", "Back": "A Rust rule"}))
        .bind(Some("Front\x1fBack"))
        .bind("rust ownership borrowing rules")
        .bind(1_i64)
        .bind(0_i32)
        .bind(2_i64)
        .bind(100_i64)
        .bind(vec!["postgres".to_string()])
        .bind(serde_json::json!({"Front": "What is SQLx?", "Back": "A Rust SQL toolkit"}))
        .bind(Some("Front\x1fBack"))
        .bind("postgres sqlx migrations testing")
        .bind(1_i64)
        .bind(0_i32)
        .execute(pool)
        .await
        .unwrap();

        sqlx::query(
            "INSERT INTO cards \
             (card_id, note_id, deck_id, ord, ivl, lapses, reps, queue, type, mtime, usn) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11), \
                    ($12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22)",
        )
        .bind(1000_i64)
        .bind(1_i64)
        .bind(10_i64)
        .bind(0_i32)
        .bind(30_i32)
        .bind(1_i32)
        .bind(5_i32)
        .bind(0_i32)
        .bind(0_i32)
        .bind(1_i64)
        .bind(0_i32)
        .bind(2000_i64)
        .bind(2_i64)
        .bind(20_i64)
        .bind(0_i32)
        .bind(5_i32)
        .bind(0_i32)
        .bind(2_i32)
        .bind(0_i32)
        .bind(0_i32)
        .bind(1_i64)
        .bind(0_i32)
        .execute(pool)
        .await
        .unwrap();
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
        let _svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            fake_repo(),
            false,
            20,
        );
    }

    #[tokio::test]
    async fn service_new_without_reranker() {
        let _svc: SearchService<FakeEmbedding, FakeVectorRepo, FakeReranker> = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            None,
            fake_repo(),
            false,
            20,
        );
    }

    // ── SearchService::search tests ──────────────────────────────

    #[tokio::test]
    async fn search_empty_query_returns_empty() {
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            fake_repo(),
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
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            fake_repo(),
            false,
            20,
        );

        let result = svc.search(&params("   ")).await.unwrap();

        assert!(result.results.is_empty());
    }

    #[tokio::test]
    async fn search_semantic_only_skips_fts() {
        let semantic_results = vec![(1, 0.95_f32), (2, 0.85)];
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(semantic_results),
            Some(FakeReranker::new()),
            fake_repo(),
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
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            Arc::new(FakeSearchReadRepository::failing_lexical()),
            false,
            20,
        );

        let result = svc.search(&params_fts_only("test query")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn search_fts_only_returns_hits_with_real_pool() {
        let Some((pool, _container)) = setup_real_pool().await else {
            return;
        };
        seed_search_fixture(&pool).await;

        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            Arc::new(SqlxSearchReadRepository::new(pool.clone())),
            false,
            20,
        );

        let result = svc.search(&params_fts_only("ownership")).await.unwrap();

        assert_eq!(result.lexical_mode, LexicalMode::Fts);
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].note_id, 1);
        assert!(result.results[0].headline.is_some());
    }

    #[tokio::test]
    async fn search_returns_query_in_result() {
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(vec![(1, 0.9)]),
            Some(FakeReranker::new()),
            fake_repo(),
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
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(vec![(1, 0.9)]),
            Some(FakeReranker::new()),
            Arc::new(FakeSearchReadRepository::with_note_details(&[1])),
            false, // rerank_enabled = false
            20,
        );

        let result = svc.search(&params_semantic_only("test")).await.unwrap();

        assert!(!result.rerank_applied);
        assert!(result.rerank_model.is_none());
        assert!(result.rerank_top_n.is_none());
    }

    #[tokio::test]
    async fn search_rerank_failure_degrades_gracefully() {
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(vec![(1, 0.95), (2, 0.85), (3, 0.75)]),
            Some(FakeReranker::failing()),
            Arc::new(FakeSearchReadRepository::with_note_details(&[1, 2, 3])),
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
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(vec![(1, 0.95), (2, 0.85)]),
            Some(FakeReranker::new()),
            Arc::new(FakeSearchReadRepository::with_note_details(&[1, 2])),
            false, // rerank_enabled = false globally
            20,
        );

        // Override with rerank_override=Some(true) should enable reranking
        let result = svc
            .search(&SearchParams {
                query: "test query".into(),
                search_mode: SearchMode::SemanticOnly,
                rerank_override: Some(true),
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(result.query, "test query");
        assert!(result.rerank_applied);
        assert_eq!(result.rerank_model.as_deref(), Some("fake/reranker"));
    }

    #[tokio::test]
    async fn search_rerank_override_disables_reranking() {
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(vec![(1, 0.95)]),
            Some(FakeReranker::new()),
            Arc::new(FakeSearchReadRepository::with_note_details(&[1])),
            true, // rerank_enabled = true globally
            20,
        );

        // Override with rerank_override=Some(false) should disable reranking
        let result = svc
            .search(&SearchParams {
                query: "test query".into(),
                search_mode: SearchMode::SemanticOnly,
                rerank_override: Some(false),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(!result.rerank_applied);
    }

    #[tokio::test]
    async fn search_no_reranker_with_rerank_enabled_degrades() {
        let svc: SearchService<FakeEmbedding, FakeVectorRepo, FakeReranker> = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(vec![(1, 0.95)]),
            None, // no reranker provided
            Arc::new(FakeSearchReadRepository::with_note_details(&[1])),
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
        let many_results: Vec<(i64, f32)> = (1..=20).map(|i| (i, 1.0 - i as f32 * 0.01)).collect();
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(many_results),
            Some(FakeReranker::new()),
            fake_repo(),
            false,
            20,
        );

        let result = svc
            .search(&SearchParams {
                query: "test query".into(),
                limit: 5,
                search_mode: SearchMode::SemanticOnly,
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(result.results.len() <= 5);
    }

    #[tokio::test]
    async fn search_rerank_uses_note_text_from_real_pool() {
        let Some((pool, _container)) = setup_real_pool().await else {
            return;
        };
        seed_search_fixture(&pool).await;

        let seen_documents = Arc::new(Mutex::new(Vec::new()));
        let reranker = RecordingReranker::new(Arc::clone(&seen_documents));
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::new(vec![(1, 0.95), (2, 0.85)]),
            Some(reranker),
            Arc::new(SqlxSearchReadRepository::new(pool.clone())),
            true,
            2,
        );

        let result = svc
            .search(&SearchParams {
                query: "rust ownership".into(),
                search_mode: SearchMode::SemanticOnly,
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(result.rerank_applied);
        let recorded_documents = seen_documents.lock().unwrap().clone();
        assert_eq!(recorded_documents.len(), 2);
        assert_eq!(
            recorded_documents[0],
            (1, "rust ownership borrowing rules".to_string())
        );
        assert_eq!(
            recorded_documents[1],
            (2, "postgres sqlx migrations testing".to_string())
        );
    }

    // ── Send + Sync ──────────────────────────────────────────────

    common::assert_send_sync!(
        SearchService<FakeEmbedding, FakeVectorRepo, FakeReranker>,
        HybridSearchResult,
        NoteDetail,
    );

    // ── get_notes_details tests ─────────────────────────────────

    #[tokio::test]
    async fn get_notes_details_empty_ids_returns_empty_map() {
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            fake_repo(),
            false,
            20,
        );

        let result = svc.get_notes_details(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn get_notes_details_with_ids_returns_db_error() {
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            Arc::new(FakeSearchReadRepository::failing_note_details()),
            false,
            20,
        );

        let result = svc.get_notes_details(&[1, 2, 3]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn get_notes_details_returns_real_rows() {
        let Some((pool, _container)) = setup_real_pool().await else {
            return;
        };
        seed_search_fixture(&pool).await;

        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            Arc::new(SqlxSearchReadRepository::new(pool.clone())),
            false,
            20,
        );

        let result = svc.get_notes_details(&[1, 2]).await.unwrap();

        assert_eq!(result[&1].normalized_text, "rust ownership borrowing rules");
        assert_eq!(result[&1].deck_names, vec!["Default".to_string()]);
        assert!(result[&1].mature);
        assert_eq!(result[&2].deck_names, vec!["Archive".to_string()]);
    }

    #[tokio::test]
    async fn search_fts_only_uses_repository_results() {
        let lexical = vec![FtsResult {
            note_id: 7,
            rank: 0.8,
            headline: Some("repository headline".to_string()),
            source: FtsSource::Fts,
        }];
        let svc = SearchService::new(
            FakeEmbedding,
            FakeVectorRepo::empty(),
            Some(FakeReranker::new()),
            Arc::new(FakeSearchReadRepository::with_lexical(
                lexical,
                LexicalMode::Fts,
            )),
            false,
            20,
        );

        let result = svc.search(&params_fts_only("ownership")).await.unwrap();

        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].note_id, 7);
        assert_eq!(
            result.results[0].headline.as_deref(),
            Some("repository headline")
        );
    }
}

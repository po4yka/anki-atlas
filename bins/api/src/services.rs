use std::env;
use std::sync::Arc;

use analytics::AnalyticsError;
use analytics::coverage::{TopicCoverage, TopicGap, WeakNote};
use analytics::duplicates::{DuplicateCluster, DuplicateStats};
use analytics::service::AnalyticsService;
use anyhow::{Context, Result as AnyhowResult};
use common::config::{ApiSettings, EmbeddingProviderKind, Settings};
use database::create_pool;
use indexer::embeddings::{EmbeddingProvider, EmbeddingProviderConfig, create_embedding_provider};
use indexer::qdrant::{
    NotePayload, QdrantRepository, SearchFilters, SparseVector, VectorRepository, VectorStoreError,
};
use jobs::{JobManager, RedisJobManager};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    Condition, Filter, Range, RecommendPointsBuilder, SearchPointsBuilder, point_id,
};
use search::error::SearchError;
use search::reranker::{CrossEncoderReranker, Reranker};
use search::service::{HybridSearchResult, SearchParams, SearchService};
use serde_json::Value;
use sqlx::PgPool;

use crate::state::AppState;

type SharedEmbeddingProvider = Arc<dyn EmbeddingProvider>;
type SharedVectorRepository = Arc<dyn VectorRepository>;
type SharedReranker = Arc<dyn Reranker>;

struct ApiVectorRepository {
    _inner: QdrantRepository,
    client: Qdrant,
    collection_name: String,
}

impl ApiVectorRepository {
    fn new(
        inner: QdrantRepository,
        url: &str,
        collection_name: &str,
    ) -> Result<Self, VectorStoreError> {
        let client = Qdrant::from_url(url)
            .build()
            .map_err(|error| VectorStoreError::Connection(error.to_string()))?;

        Ok(Self {
            _inner: inner,
            client,
            collection_name: collection_name.to_string(),
        })
    }

    fn unsupported_error(&self, operation: &str) -> VectorStoreError {
        VectorStoreError::Client(format!(
            "API vector adapter does not implement `{operation}` because write-side Qdrant operations still belong in indexer"
        ))
    }

    fn build_filters(&self, filters: &SearchFilters) -> Option<Filter> {
        let mut must = Vec::new();
        let mut must_not = Vec::new();

        if let Some(deck_names) = filters.deck_names.clone().filter(|items| !items.is_empty()) {
            must.push(Condition::matches("deck_names", deck_names));
        }
        if let Some(tags) = filters.tags.clone().filter(|items| !items.is_empty()) {
            must.push(Condition::matches("tags", tags));
        }
        if let Some(model_ids) = filters.model_ids.clone().filter(|items| !items.is_empty()) {
            must.push(Condition::matches("model_id", model_ids));
        }
        if filters.mature_only {
            must.push(Condition::matches("mature", true));
        }
        if let Some(min_reps) = filters.min_reps {
            must.push(Condition::range(
                "reps",
                Range {
                    gte: Some(f64::from(min_reps)),
                    ..Default::default()
                },
            ));
        }
        if let Some(max_lapses) = filters.max_lapses {
            must.push(Condition::range(
                "lapses",
                Range {
                    lte: Some(f64::from(max_lapses)),
                    ..Default::default()
                },
            ));
        }
        if let Some(deck_names_exclude) = filters
            .deck_names_exclude
            .clone()
            .filter(|items| !items.is_empty())
        {
            must_not.push(Condition::matches("deck_names", deck_names_exclude));
        }
        if let Some(tags_exclude) = filters
            .tags_exclude
            .clone()
            .filter(|items| !items.is_empty())
        {
            must_not.push(Condition::matches("tags", tags_exclude));
        }

        if must.is_empty() && must_not.is_empty() {
            None
        } else if must.is_empty() && !must_not.is_empty() {
            Some(Filter::must_not(must_not))
        } else if !must.is_empty() && must_not.is_empty() {
            Some(Filter::must(must))
        } else if !must.is_empty() && !must_not.is_empty() {
            Some(Filter {
                must,
                should: Vec::new(),
                must_not,
                min_should: None,
            })
        } else {
            None
        }
    }

    fn extract_note_id(
        &self,
        id: Option<qdrant_client::qdrant::PointId>,
    ) -> Result<i64, VectorStoreError> {
        match id.and_then(|point_id| point_id.point_id_options) {
            Some(point_id::PointIdOptions::Num(value)) => i64::try_from(value).map_err(|_| {
                VectorStoreError::Client(format!("point id {value} does not fit into i64"))
            }),
            Some(point_id::PointIdOptions::Uuid(value)) => Err(VectorStoreError::Client(format!(
                "UUID point ids are not supported for note-backed API search results: {value}"
            ))),
            None => Err(VectorStoreError::Client(
                "Qdrant search result was missing a point id".to_string(),
            )),
        }
    }
}

#[async_trait::async_trait]
impl VectorRepository for ApiVectorRepository {
    async fn ensure_collection(&self, _dimension: usize) -> Result<bool, VectorStoreError> {
        Err(self.unsupported_error("ensure_collection"))
    }

    async fn upsert_vectors(
        &self,
        _vectors: &[Vec<f32>],
        _payloads: &[NotePayload],
        _sparse_vectors: Option<&[SparseVector]>,
    ) -> Result<usize, VectorStoreError> {
        Err(self.unsupported_error("upsert_vectors"))
    }

    async fn delete_vectors(&self, _note_ids: &[i64]) -> Result<usize, VectorStoreError> {
        Err(self.unsupported_error("delete_vectors"))
    }

    async fn get_existing_hashes(
        &self,
        _note_ids: &[i64],
    ) -> Result<std::collections::HashMap<i64, String>, VectorStoreError> {
        Err(self.unsupported_error("get_existing_hashes"))
    }

    async fn search(
        &self,
        query_vector: &[f32],
        _query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<(i64, f32)>, VectorStoreError> {
        let mut request =
            SearchPointsBuilder::new(&self.collection_name, query_vector.to_vec(), limit as u64);
        if let Some(filter) = self.build_filters(filters) {
            request = request.filter(filter);
        }

        let response = self
            .client
            .search_points(request)
            .await
            .map_err(|error| VectorStoreError::Client(error.to_string()))?;

        response
            .result
            .into_iter()
            .map(|point| Ok((self.extract_note_id(point.id)?, point.score)))
            .collect()
    }

    async fn find_similar_to_note(
        &self,
        note_id: i64,
        limit: usize,
        min_score: f32,
        deck_names: Option<&[String]>,
        tags: Option<&[String]>,
    ) -> Result<Vec<(i64, f32)>, VectorStoreError> {
        let filters = SearchFilters {
            deck_names: deck_names.map(|items| items.to_vec()),
            tags: tags.map(|items| items.to_vec()),
            ..Default::default()
        };
        let mut request = RecommendPointsBuilder::new(&self.collection_name, (limit + 1) as u64)
            .add_positive(u64::try_from(note_id).map_err(|_| {
                VectorStoreError::Client(format!(
                    "note id {note_id} cannot be represented as a Qdrant numeric point id"
                ))
            })?)
            .score_threshold(min_score);
        if let Some(filter) = self.build_filters(&filters) {
            request = request.filter(filter);
        }

        let response = self
            .client
            .recommend(request)
            .await
            .map_err(|error| VectorStoreError::Client(error.to_string()))?;

        let mut results = Vec::new();
        for point in response.result {
            let found_note_id = self.extract_note_id(point.id)?;
            if found_note_id != note_id {
                results.push((found_note_id, point.score));
            }
            if results.len() == limit {
                break;
            }
        }

        Ok(results)
    }

    async fn close(&self) -> Result<(), VectorStoreError> {
        Ok(())
    }
}

#[async_trait::async_trait]
pub trait SearchFacade: Send + Sync {
    async fn search(
        &self,
        params: &SearchParams,
    ) -> std::result::Result<HybridSearchResult, SearchError>;
}

#[async_trait::async_trait]
pub trait AnalyticsFacade: Send + Sync {
    async fn get_taxonomy_tree(
        &self,
        root_path: Option<String>,
    ) -> std::result::Result<Vec<Value>, AnalyticsError>;
    async fn get_coverage(
        &self,
        topic_path: String,
        include_subtree: bool,
    ) -> std::result::Result<Option<TopicCoverage>, AnalyticsError>;
    async fn get_gaps(
        &self,
        topic_path: String,
        min_coverage: i64,
    ) -> std::result::Result<Vec<TopicGap>, AnalyticsError>;
    async fn get_weak_notes(
        &self,
        topic_path: String,
        max_results: i64,
    ) -> std::result::Result<Vec<WeakNote>, AnalyticsError>;
    async fn find_duplicates(
        &self,
        threshold: f64,
        max_clusters: usize,
        deck_filter: Option<Vec<String>>,
        tag_filter: Option<Vec<String>>,
    ) -> std::result::Result<(Vec<DuplicateCluster>, DuplicateStats), AnalyticsError>;
}

struct SearchFacadeImpl {
    inner: SearchService<SharedEmbeddingProvider, SharedVectorRepository, SharedReranker>,
}

#[async_trait::async_trait]
impl SearchFacade for SearchFacadeImpl {
    async fn search(
        &self,
        params: &SearchParams,
    ) -> std::result::Result<HybridSearchResult, SearchError> {
        self.inner.search(params).await
    }
}

struct AnalyticsFacadeImpl {
    inner: AnalyticsService<SharedEmbeddingProvider, SharedVectorRepository>,
}

#[async_trait::async_trait]
impl AnalyticsFacade for AnalyticsFacadeImpl {
    async fn get_taxonomy_tree(
        &self,
        root_path: Option<String>,
    ) -> std::result::Result<Vec<Value>, AnalyticsError> {
        self.inner.get_taxonomy_tree(root_path.as_deref()).await
    }

    async fn get_coverage(
        &self,
        topic_path: String,
        include_subtree: bool,
    ) -> std::result::Result<Option<TopicCoverage>, AnalyticsError> {
        self.inner.get_coverage(&topic_path, include_subtree).await
    }

    async fn get_gaps(
        &self,
        topic_path: String,
        min_coverage: i64,
    ) -> std::result::Result<Vec<TopicGap>, AnalyticsError> {
        self.inner.get_gaps(&topic_path, min_coverage).await
    }

    async fn get_weak_notes(
        &self,
        topic_path: String,
        max_results: i64,
    ) -> std::result::Result<Vec<WeakNote>, AnalyticsError> {
        self.inner.get_weak_notes(&topic_path, max_results).await
    }

    async fn find_duplicates(
        &self,
        threshold: f64,
        max_clusters: usize,
        deck_filter: Option<Vec<String>>,
        tag_filter: Option<Vec<String>>,
    ) -> std::result::Result<(Vec<DuplicateCluster>, DuplicateStats), AnalyticsError> {
        self.inner
            .find_duplicates(
                threshold,
                max_clusters,
                deck_filter.as_deref(),
                tag_filter.as_deref(),
            )
            .await
    }
}

/// Shared runtime services used by HTTP handlers.
pub struct ApiServices {
    pub db: PgPool,
    pub job_manager: Arc<dyn JobManager>,
    pub search: Arc<dyn SearchFacade>,
    pub analytics: Arc<dyn AnalyticsFacade>,
}

impl ApiServices {
    pub fn new(
        db: PgPool,
        job_manager: Arc<dyn JobManager>,
        search: Arc<dyn SearchFacade>,
        analytics: Arc<dyn AnalyticsFacade>,
    ) -> Self {
        Self {
            db,
            job_manager,
            search,
            analytics,
        }
    }
}

fn build_embedding_config(settings: &Settings) -> AnyhowResult<EmbeddingProviderConfig> {
    let embedding = settings.embedding();
    let config = match embedding.provider {
        EmbeddingProviderKind::Mock => EmbeddingProviderConfig::Mock {
            dimension: embedding.dimension as usize,
        },
        EmbeddingProviderKind::OpenAi => EmbeddingProviderConfig::OpenAi {
            model: embedding.model,
            dimension: embedding.dimension as usize,
            batch_size: None,
            api_key: env::var("OPENAI_API_KEY")
                .context("OPENAI_API_KEY must be set for the OpenAI embedding provider")?,
        },
        EmbeddingProviderKind::Google => EmbeddingProviderConfig::Google {
            model: embedding.model,
            dimension: embedding.dimension as usize,
            batch_size: None,
            api_key: env::var("GOOGLE_API_KEY")
                .context("GOOGLE_API_KEY must be set for the Google embedding provider")?,
        },
    };

    Ok(config)
}

fn build_reranker(settings: &Settings) -> Option<SharedReranker> {
    let rerank = settings.rerank();
    if !rerank.enabled {
        return None;
    }

    let endpoint = env::var("ANKIATLAS_RERANK_ENDPOINT")
        .ok()
        .filter(|value| !value.is_empty());
    let endpoint = match endpoint {
        Some(endpoint) => endpoint,
        None => {
            tracing::warn!(
                "reranking enabled in settings but ANKIATLAS_RERANK_ENDPOINT is not set; disabling reranker"
            );
            return None;
        }
    };

    Some(Arc::new(CrossEncoderReranker::new(
        rerank.model,
        rerank.batch_size as usize,
        endpoint,
    )))
}

pub async fn build_api_services(settings: &Settings) -> AnyhowResult<ApiServices> {
    let db = create_pool(&settings.database())
        .await
        .context("create PostgreSQL pool for API")?;
    let embedding: SharedEmbeddingProvider = Arc::from(
        create_embedding_provider(&build_embedding_config(settings)?)
            .context("create embedding provider for API")?,
    );
    let collection_name = "anki_notes";
    let vector_repo = Arc::new(
        ApiVectorRepository::new(
            QdrantRepository::new(&settings.qdrant_url, collection_name)
                .await
                .context("connect Qdrant repository for API")?,
            &settings.qdrant_url,
            collection_name,
        )
        .context("build API vector adapter")?,
    ) as SharedVectorRepository;
    let reranker = build_reranker(settings);
    let rerank_enabled = settings.rerank().enabled && reranker.is_some();

    let search = Arc::new(SearchFacadeImpl {
        inner: SearchService::new(
            embedding.clone(),
            vector_repo.clone(),
            reranker,
            db.clone(),
            rerank_enabled,
            settings.rerank().top_n as usize,
        ),
    }) as Arc<dyn SearchFacade>;

    let analytics = Arc::new(AnalyticsFacadeImpl {
        inner: AnalyticsService::new(embedding, vector_repo, db.clone()),
    }) as Arc<dyn AnalyticsFacade>;

    let job_settings = settings.jobs();
    let job_manager = Arc::new(
        RedisJobManager::new(
            &job_settings.redis_url,
            &job_settings.queue_name,
            job_settings.max_retries,
            u64::from(job_settings.result_ttl_seconds),
        )
        .await
        .context("create Redis job manager for API")?,
    ) as Arc<dyn JobManager>;

    Ok(ApiServices::new(db, job_manager, search, analytics))
}

pub fn build_app_state(api_settings: ApiSettings, services: ApiServices) -> AppState {
    AppState {
        api: Arc::new(api_settings),
        services: Arc::new(services),
    }
}

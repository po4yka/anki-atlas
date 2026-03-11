use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use analytics::AnalyticsError;
use analytics::coverage::{TopicCoverage, TopicGap, WeakNote};
use analytics::duplicates::{DuplicateCluster, DuplicateStats};
use analytics::labeling::LabelingStats;
use analytics::service::AnalyticsService;
use analytics::taxonomy::Taxonomy;
use anyhow::{Context, Result as AnyhowResult};
use common::config::{EmbeddingProviderKind, Settings, qdrant_grpc_url};
use database::create_pool;
use indexer::embeddings::{EmbeddingProvider, EmbeddingProviderConfig, create_embedding_provider};
use indexer::qdrant::VectorRepository;
use jobs::{JobManager, RedisJobManager};
use qdrant_client::Qdrant;
use search::error::SearchError;
use search::reranker::{CrossEncoderReranker, Reranker};
use search::service::{HybridSearchResult, SearchParams, SearchService};
use serde_json::Value;
use sqlx::PgPool;

use crate::workflows::{
    GeneratePreviewService, IndexExecutor, IndexingService, ObsidianScanService, QdrantVectorStore,
    SyncExecutionService, SyncExecutor, TagAuditService, ValidationService,
};

type SharedEmbeddingProvider = Arc<dyn EmbeddingProvider>;
type SharedVectorRepository = Arc<dyn VectorRepository>;
type SharedReranker = Arc<dyn Reranker>;

#[derive(Debug, Clone, Copy, Default)]
pub struct BuildSurfaceServicesOptions {
    pub enable_direct_execution: bool,
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
    async fn load_taxonomy(
        &self,
        yaml_path: Option<PathBuf>,
    ) -> std::result::Result<Taxonomy, AnalyticsError>;
    async fn label_notes(
        &self,
        yaml_path: Option<PathBuf>,
        min_confidence: f32,
    ) -> std::result::Result<LabelingStats, AnalyticsError>;
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
    async fn load_taxonomy(
        &self,
        yaml_path: Option<PathBuf>,
    ) -> std::result::Result<Taxonomy, AnalyticsError> {
        self.inner.load_taxonomy(yaml_path.as_deref()).await
    }

    async fn label_notes(
        &self,
        yaml_path: Option<PathBuf>,
        min_confidence: f32,
    ) -> std::result::Result<LabelingStats, AnalyticsError> {
        let taxonomy = if let Some(path) = yaml_path {
            Some(self.inner.load_taxonomy(Some(&path)).await?)
        } else {
            None
        };
        self.inner
            .label_notes(taxonomy.as_ref(), min_confidence)
            .await
    }

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

pub struct SurfaceServices {
    pub db: PgPool,
    pub job_manager: Arc<dyn JobManager>,
    pub search: Arc<dyn SearchFacade>,
    pub analytics: Arc<dyn AnalyticsFacade>,
    pub sync: Arc<dyn SyncExecutor>,
    pub index: Arc<dyn IndexExecutor>,
    pub generate_preview: Arc<GeneratePreviewService>,
    pub validation: Arc<ValidationService>,
    pub obsidian_scan: Arc<ObsidianScanService>,
    pub tag_audit: Arc<TagAuditService>,
}

impl SurfaceServices {
    pub fn new(
        db: PgPool,
        job_manager: Arc<dyn JobManager>,
        search: Arc<dyn SearchFacade>,
        analytics: Arc<dyn AnalyticsFacade>,
    ) -> Self {
        Self {
            sync: Arc::new(SyncExecutionService::unsupported(db.clone())),
            index: Arc::new(IndexingService::unsupported(db.clone())),
            generate_preview: Arc::new(GeneratePreviewService::new()),
            validation: Arc::new(ValidationService::new()),
            obsidian_scan: Arc::new(ObsidianScanService::new()),
            tag_audit: Arc::new(TagAuditService::new()),
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

pub async fn build_surface_services(
    settings: &Settings,
    options: BuildSurfaceServicesOptions,
) -> AnyhowResult<SurfaceServices> {
    let db = create_pool(&settings.database())
        .await
        .context("create PostgreSQL pool for surface runtime")?;
    let embedding: SharedEmbeddingProvider = Arc::from(
        create_embedding_provider(&build_embedding_config(settings)?)
            .context("create embedding provider for surface runtime")?,
    );
    let collection_name = "anki_notes";
    let qdrant_client = Qdrant::from_url(&qdrant_grpc_url(&settings.qdrant_url)?)
        .build()
        .context("connect Qdrant client for surface runtime")?;
    qdrant_client
        .health_check()
        .await
        .context("check Qdrant health for surface runtime")?;

    let vector_store = Arc::new(QdrantVectorStore::new(
        qdrant_client.clone(),
        collection_name,
    ));
    vector_store
        .ensure_collection(settings.embedding().dimension as usize)
        .await
        .context("ensure Qdrant collection for surface runtime")?;
    let vector_repo = vector_store as SharedVectorRepository;
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
        inner: AnalyticsService::new(embedding.clone(), vector_repo.clone(), db.clone()),
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
        .context("create Redis job manager for surface runtime")?,
    ) as Arc<dyn JobManager>;

    let mut services = SurfaceServices::new(db.clone(), job_manager, search, analytics);
    if options.enable_direct_execution {
        let index = Arc::new(IndexingService::new(db.clone(), embedding, vector_repo));
        services.sync = Arc::new(SyncExecutionService::new(db, index.clone()));
        services.index = index;
    }

    Ok(services)
}

use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use analytics::AnalyticsError;
use analytics::coverage::{TopicCoverage, TopicGap, WeakNote};
use analytics::duplicates::{DuplicateCluster, DuplicateStats};
use analytics::labeling::LabelingStats;
use analytics::repository::SqlxAnalyticsRepository;
use analytics::service::AnalyticsService;
use analytics::taxonomy::Taxonomy;
use anyhow::{Context, Result as AnyhowResult, anyhow};
use common::config::{EmbeddingProviderKind, Settings};
use database::create_pool;
use indexer::embeddings::{EmbeddingProvider, EmbeddingProviderConfig, create_embedding_provider};
use indexer::qdrant::{QdrantRepository, VectorRepository};
use jobs::{JobManager, RedisJobManager};
use search::error::SearchError;
use search::repository::SqlxSearchReadRepository;
use search::reranker::{CrossEncoderReranker, Reranker};
use search::service::{
    ChunkSearchParams, ChunkSearchResult, HybridSearchResult, SearchParams, SearchService,
};
use serde_json::Value;
use sqlx::PgPool;

use crate::workflows::{
    GeneratePreviewService, IndexExecutor, IndexingService, ObsidianScanService,
    SyncExecutionService, TagAuditService, ValidationService,
};

type SharedEmbeddingProvider = Arc<dyn EmbeddingProvider>;
type SharedVectorRepository = Arc<dyn VectorRepository>;
type SharedReranker = Arc<dyn Reranker>;

const EMBEDDING_VECTOR_SCHEMA: &str = "multimodal_v1";

#[derive(Debug, Clone, Copy, Default)]
pub struct BuildSurfaceServicesOptions {
    pub enable_direct_execution: bool,
}

#[async_trait::async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait SearchFacade: Send + Sync {
    async fn search(
        &self,
        params: &SearchParams,
    ) -> std::result::Result<HybridSearchResult, SearchError>;

    async fn search_chunks(
        &self,
        params: &ChunkSearchParams,
    ) -> std::result::Result<ChunkSearchResult, SearchError>;
}

#[async_trait::async_trait]
#[cfg_attr(test, mockall::automock)]
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

    async fn search_chunks(
        &self,
        params: &ChunkSearchParams,
    ) -> std::result::Result<ChunkSearchResult, SearchError> {
        self.inner.search_chunks(params).await
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
    pub sync: Arc<SyncExecutionService>,
    pub index: Arc<dyn IndexExecutor>,
    pub generate_preview: Arc<GeneratePreviewService>,
    pub validation: Arc<ValidationService>,
    pub obsidian_scan: Arc<ObsidianScanService>,
    pub tag_audit: Arc<TagAuditService>,
    direct_execution_enabled: bool,
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
            direct_execution_enabled: false,
            db,
            job_manager,
            search,
            analytics,
        }
    }

    #[cfg(test)]
    pub(crate) fn direct_execution_enabled(&self) -> bool {
        self.direct_execution_enabled
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
            api_key: env::var("GEMINI_API_KEY")
                .or_else(|_| env::var("GOOGLE_API_KEY"))
                .context(
                    "GEMINI_API_KEY or GOOGLE_API_KEY must be set for the Google embedding provider",
                )?,
        },
    };

    Ok(config)
}

async fn load_sync_metadata_value(db: &PgPool, key: &str) -> AnyhowResult<Option<String>> {
    sqlx::query_scalar::<_, String>("SELECT value #>> '{}' FROM sync_metadata WHERE key = $1")
        .bind(key)
        .fetch_optional(db)
        .await
        .with_context(|| format!("load sync metadata key `{key}`"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EmbeddingFingerprint {
    model: String,
    dimension: usize,
    vector_schema: String,
}

async fn load_embedding_fingerprint(db: &PgPool) -> AnyhowResult<Option<EmbeddingFingerprint>> {
    let model = load_sync_metadata_value(db, "embedding_model").await?;
    let dimension = load_sync_metadata_value(db, "embedding_dimension").await?;
    let vector_schema = load_sync_metadata_value(db, "embedding_vector_schema").await?;

    let (Some(model), Some(dimension), Some(vector_schema)) = (model, dimension, vector_schema)
    else {
        return Ok(None);
    };

    let dimension = dimension.parse::<usize>().with_context(|| {
        format!("parse sync metadata `embedding_dimension` value `{dimension}`")
    })?;

    Ok(Some(EmbeddingFingerprint {
        model,
        dimension,
        vector_schema,
    }))
}

fn validate_read_only_collection_state(
    current_dimension: Option<usize>,
    desired_dimension: usize,
    desired_model: &str,
    stored_fingerprint: Option<&EmbeddingFingerprint>,
) -> AnyhowResult<()> {
    let Some(current_dimension) = current_dimension else {
        if stored_fingerprint.is_some() {
            return Err(anyhow!("reindex required: vector collection is missing"));
        }
        return Ok(());
    };

    if current_dimension != desired_dimension {
        return Err(anyhow!(
            "reindex required: vector collection dimension is {current_dimension}, expected {desired_dimension}"
        ));
    }

    let Some(stored_fingerprint) = stored_fingerprint else {
        return Ok(());
    };

    if stored_fingerprint.model != desired_model {
        return Err(anyhow!(
            "reindex required: stored embedding model is {}, current model is {desired_model}",
            stored_fingerprint.model
        ));
    }
    if stored_fingerprint.dimension != desired_dimension {
        return Err(anyhow!(
            "reindex required: stored embedding dimension is {}, current dimension is {desired_dimension}",
            stored_fingerprint.dimension
        ));
    }
    if stored_fingerprint.vector_schema != EMBEDDING_VECTOR_SCHEMA {
        return Err(anyhow!(
            "reindex required: stored vector schema is {}, expected {EMBEDDING_VECTOR_SCHEMA}",
            stored_fingerprint.vector_schema
        ));
    }

    Ok(())
}

async fn validate_read_only_vector_store(
    db: &PgPool,
    vector_store: &QdrantRepository,
    embedding: &dyn EmbeddingProvider,
) -> AnyhowResult<()> {
    let desired_dimension = embedding.dimension();
    let desired_model = embedding.model_name();
    let current_dimension = vector_store
        .collection_dimension()
        .await
        .context("inspect Qdrant collection dimension")?;
    let stored_fingerprint = load_embedding_fingerprint(db).await?;

    validate_read_only_collection_state(
        current_dimension,
        desired_dimension,
        desired_model,
        stored_fingerprint.as_ref(),
    )
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
    let vector_store = Arc::new(
        QdrantRepository::new(&settings.qdrant_url, collection_name)
            .await
            .context("connect Qdrant repository for surface runtime")?,
    );
    if !options.enable_direct_execution {
        validate_read_only_vector_store(&db, &vector_store, embedding.as_ref()).await?;
    }
    let vector_repo = vector_store as SharedVectorRepository;
    let reranker = build_reranker(settings);
    let rerank_enabled = settings.rerank().enabled && reranker.is_some();

    let search = Arc::new(SearchFacadeImpl {
        inner: SearchService::new(
            embedding.clone(),
            vector_repo.clone(),
            reranker,
            Arc::new(SqlxSearchReadRepository::new(db.clone())),
            rerank_enabled,
            settings.rerank().top_n as usize,
        ),
    }) as Arc<dyn SearchFacade>;

    let analytics = Arc::new(AnalyticsFacadeImpl {
        inner: AnalyticsService::new(
            embedding.clone(),
            vector_repo.clone(),
            Arc::new(SqlxAnalyticsRepository::new(db.clone())),
        ),
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
        let index = Arc::new(IndexingService::new(
            db.clone(),
            embedding,
            vector_repo,
            settings.anki_collection_path.as_ref().map(PathBuf::from),
            settings.anki_media_root.as_ref().map(PathBuf::from),
        ));
        services.sync = Arc::new(SyncExecutionService::new(db, index.clone()));
        services.index = index;
        services.direct_execution_enabled = true;
    }

    Ok(services)
}

#[cfg(test)]
mod tests {
    use super::{
        AnalyticsFacade, EMBEDDING_VECTOR_SCHEMA, EmbeddingFingerprint, SearchFacade,
        SurfaceServices, validate_read_only_collection_state,
    };
    use crate::services::{MockAnalyticsFacade, MockSearchFacade};
    use jobs::JobManager;
    use sqlx::postgres::PgPoolOptions;
    use std::sync::Arc;

    struct NoopJobManager;

    #[async_trait::async_trait]
    impl JobManager for NoopJobManager {
        async fn enqueue_sync_job(
            &self,
            _payload: jobs::types::SyncJobPayload,
            _run_at: Option<chrono::DateTime<chrono::Utc>>,
        ) -> Result<jobs::types::JobRecord, jobs::JobError> {
            unreachable!("job manager is not exercised in this test")
        }

        async fn enqueue_index_job(
            &self,
            _payload: jobs::types::IndexJobPayload,
            _run_at: Option<chrono::DateTime<chrono::Utc>>,
        ) -> Result<jobs::types::JobRecord, jobs::JobError> {
            unreachable!("job manager is not exercised in this test")
        }

        async fn get_job(&self, _job_id: &str) -> Result<jobs::types::JobRecord, jobs::JobError> {
            unreachable!("job manager is not exercised in this test")
        }

        async fn cancel_job(
            &self,
            _job_id: &str,
        ) -> Result<jobs::types::JobRecord, jobs::JobError> {
            unreachable!("job manager is not exercised in this test")
        }

        async fn close(&self) -> Result<(), jobs::JobError> {
            Ok(())
        }
    }

    #[test]
    fn fresh_runtime_allows_missing_collection_without_fingerprint() {
        let result = validate_read_only_collection_state(None, 384, "mock/test", None);
        assert!(result.is_ok());
    }

    #[test]
    fn missing_collection_requires_reindex_when_fingerprint_exists() {
        let result = validate_read_only_collection_state(
            None,
            384,
            "mock/test",
            Some(&EmbeddingFingerprint {
                model: "mock/test".to_string(),
                dimension: 384,
                vector_schema: EMBEDDING_VECTOR_SCHEMA.to_string(),
            }),
        );

        let error = result.expect_err("missing indexed collection should fail");
        assert_eq!(
            error.to_string(),
            "reindex required: vector collection is missing"
        );
    }

    #[tokio::test]
    async fn surface_services_new_keeps_direct_execution_disabled_by_default() {
        let db = PgPoolOptions::new()
            .connect_lazy("postgres://localhost/anki_atlas")
            .expect("lazy postgres pool");
        let search = Arc::new(MockSearchFacade::new()) as Arc<dyn SearchFacade>;
        let analytics = Arc::new(MockAnalyticsFacade::new()) as Arc<dyn AnalyticsFacade>;
        let services = SurfaceServices::new(
            db,
            Arc::new(NoopJobManager) as Arc<dyn JobManager>,
            search,
            analytics,
        );

        assert!(!services.direct_execution_enabled());
    }
}

use std::path::PathBuf;
use std::sync::Arc;

use analytics::coverage::{TopicCoverage, TopicGap, WeakNote};
use analytics::duplicates::{DuplicateCluster, DuplicateStats};
use analytics::labeling::LabelingStats;
use analytics::taxonomy::Taxonomy;
use common::config::{EmbeddingProviderKind, Settings};
use database::MigrationResult;
use search::fts::SearchFilters;
use search::service::{ChunkSearchParams, ChunkSearchResult, HybridSearchResult, SearchParams};
use surface_runtime::{
    AnalyticsFacade, BuildSurfaceServicesOptions, GeneratePreview, GeneratePreviewService,
    IndexExecutionSummary, IndexExecutor, ObsidianScanPreview, ObsidianScanService, SearchFacade,
    SurfaceProgressSink, SurfaceServices, SyncExecutionHandle, SyncExecutionSummary,
    TagAuditService, TagAuditSummary, ValidationService, ValidationSummary, build_surface_services,
};

#[derive(Debug, Clone)]
pub struct RuntimeSettingsSummary {
    pub postgres_url: String,
    pub qdrant_url: String,
    pub redis_url: String,
    pub embedding_provider: String,
    pub embedding_model: String,
    pub rerank_enabled: bool,
}

#[derive(Clone)]
pub struct RuntimeBootstrap {
    pub settings: Settings,
    pub summary: RuntimeSettingsSummary,
    pub services: Arc<SurfaceServices>,
}

#[derive(Clone)]
pub struct RuntimeHandles {
    pub search: Arc<dyn SearchFacade>,
    pub analytics: Arc<dyn AnalyticsFacade>,
    pub sync: SyncExecutionHandle,
    pub index: Arc<dyn IndexExecutor>,
    pub generate_preview: Arc<GeneratePreviewService>,
    pub validation: Arc<ValidationService>,
    pub obsidian_scan: Arc<ObsidianScanService>,
    pub tag_audit: Arc<TagAuditService>,
}

impl From<&SurfaceServices> for RuntimeHandles {
    fn from(services: &SurfaceServices) -> Self {
        Self {
            search: Arc::clone(&services.search),
            analytics: Arc::clone(&services.analytics),
            sync: services.sync.handle(),
            index: Arc::clone(&services.index),
            generate_preview: Arc::clone(&services.generate_preview),
            validation: Arc::clone(&services.validation),
            obsidian_scan: Arc::clone(&services.obsidian_scan),
            tag_audit: Arc::clone(&services.tag_audit),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub query: String,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
    pub limit: usize,
    pub semantic_only: bool,
    pub fts_only: bool,
}

#[derive(Debug, Clone)]
pub struct ChunkSearchRequest {
    pub query: String,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
    pub limit: usize,
}

#[derive(Debug, Clone)]
pub struct TopicsTreeRequest {
    pub root_path: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TopicsLoadRequest {
    pub file: PathBuf,
}

#[derive(Debug, Clone)]
pub struct TopicsLabelRequest {
    pub file: Option<PathBuf>,
    pub min_confidence: f32,
}

#[derive(Debug, Clone)]
pub struct CoverageRequest {
    pub topic: String,
    pub include_subtree: bool,
}

#[derive(Debug, Clone)]
pub struct GapsRequest {
    pub topic: String,
    pub min_coverage: i64,
}

#[derive(Debug, Clone)]
pub struct WeakNotesRequest {
    pub topic: String,
    pub limit: i64,
}

#[derive(Debug, Clone)]
pub struct DuplicatesRequest {
    pub threshold: f64,
    pub max: usize,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SyncRequest {
    pub source: PathBuf,
    pub run_migrations: bool,
    pub run_index: bool,
    pub force_reindex: bool,
}

#[derive(Debug, Clone)]
pub struct IndexRequest {
    pub force_reindex: bool,
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub file: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ValidateRequest {
    pub file: PathBuf,
    pub include_quality: bool,
}

#[derive(Debug, Clone)]
pub struct ObsidianScanRequest {
    pub vault: PathBuf,
    pub source_dirs: Vec<String>,
    pub dry_run: bool,
}

#[derive(Debug, Clone)]
pub struct TagAuditRequest {
    pub file: PathBuf,
    pub apply_fixes: bool,
}

fn embedding_provider_label(provider: EmbeddingProviderKind) -> &'static str {
    match provider {
        EmbeddingProviderKind::OpenAi => "openai",
        EmbeddingProviderKind::Google => "google",
        EmbeddingProviderKind::Mock => "mock",
    }
}

pub fn summarize_settings(settings: &Settings) -> RuntimeSettingsSummary {
    RuntimeSettingsSummary {
        postgres_url: settings.postgres_url.clone(),
        qdrant_url: settings.qdrant_url.clone(),
        redis_url: settings.redis_url.clone(),
        embedding_provider: embedding_provider_label(settings.embedding_provider).to_string(),
        embedding_model: settings.embedding_model.clone(),
        rerank_enabled: settings.rerank_enabled,
    }
}

pub async fn bootstrap_runtime(enable_direct_execution: bool) -> anyhow::Result<RuntimeBootstrap> {
    let settings = Settings::load()?;
    let services = build_surface_services(
        &settings,
        BuildSurfaceServicesOptions {
            enable_direct_execution,
        },
    )
    .await?;

    Ok(RuntimeBootstrap {
        summary: summarize_settings(&settings),
        settings,
        services: Arc::new(services),
    })
}

pub async fn run_migrations(settings: &Settings) -> anyhow::Result<MigrationResult> {
    let pool = database::create_pool(&settings.database()).await?;
    database::run_migrations(&pool).await.map_err(Into::into)
}

pub async fn search(
    handles: RuntimeHandles,
    request: SearchRequest,
) -> anyhow::Result<HybridSearchResult> {
    anyhow::ensure!(
        !(request.semantic_only && request.fts_only),
        "--semantic and --fts are mutually exclusive"
    );

    let filters =
        (!request.deck_names.is_empty() || !request.tags.is_empty()).then(|| SearchFilters {
            deck_names: (!request.deck_names.is_empty()).then(|| request.deck_names.clone()),
            tags: (!request.tags.is_empty()).then(|| request.tags.clone()),
            ..Default::default()
        });
    let params = SearchParams {
        query: request.query,
        filters,
        limit: request.limit,
        semantic_weight: 1.0,
        fts_weight: 1.0,
        semantic_only: request.semantic_only,
        fts_only: request.fts_only,
        rerank_override: None,
        rerank_top_n_override: None,
    };
    handles.search.search(&params).await.map_err(Into::into)
}

pub async fn search_chunks(
    handles: RuntimeHandles,
    request: ChunkSearchRequest,
) -> anyhow::Result<ChunkSearchResult> {
    let filters =
        (!request.deck_names.is_empty() || !request.tags.is_empty()).then(|| SearchFilters {
            deck_names: (!request.deck_names.is_empty()).then(|| request.deck_names.clone()),
            tags: (!request.tags.is_empty()).then(|| request.tags.clone()),
            ..Default::default()
        });
    let params = ChunkSearchParams {
        query: request.query,
        filters,
        limit: request.limit,
    };
    handles
        .search
        .search_chunks(&params)
        .await
        .map_err(Into::into)
}

pub async fn topics_tree(
    handles: RuntimeHandles,
    request: TopicsTreeRequest,
) -> anyhow::Result<Vec<serde_json::Value>> {
    handles
        .analytics
        .get_taxonomy_tree(request.root_path)
        .await
        .map_err(Into::into)
}

pub async fn topics_load(
    handles: RuntimeHandles,
    request: TopicsLoadRequest,
) -> anyhow::Result<Taxonomy> {
    handles
        .analytics
        .load_taxonomy(Some(request.file))
        .await
        .map_err(Into::into)
}

pub async fn topics_label(
    handles: RuntimeHandles,
    request: TopicsLabelRequest,
) -> anyhow::Result<LabelingStats> {
    handles
        .analytics
        .label_notes(request.file, request.min_confidence)
        .await
        .map_err(Into::into)
}

pub async fn coverage(
    handles: RuntimeHandles,
    request: CoverageRequest,
) -> anyhow::Result<TopicCoverage> {
    let coverage = handles
        .analytics
        .get_coverage(request.topic.clone(), request.include_subtree)
        .await?;
    coverage.ok_or_else(|| anyhow::anyhow!("topic not found: {}", request.topic))
}

pub async fn gaps(handles: RuntimeHandles, request: GapsRequest) -> anyhow::Result<Vec<TopicGap>> {
    handles
        .analytics
        .get_gaps(request.topic, request.min_coverage)
        .await
        .map_err(Into::into)
}

pub async fn weak_notes(
    handles: RuntimeHandles,
    request: WeakNotesRequest,
) -> anyhow::Result<Vec<WeakNote>> {
    handles
        .analytics
        .get_weak_notes(request.topic, request.limit)
        .await
        .map_err(Into::into)
}

pub async fn duplicates(
    handles: RuntimeHandles,
    request: DuplicatesRequest,
) -> anyhow::Result<(Vec<DuplicateCluster>, DuplicateStats)> {
    handles
        .analytics
        .find_duplicates(
            request.threshold,
            request.max,
            (!request.deck_names.is_empty()).then_some(request.deck_names),
            (!request.tags.is_empty()).then_some(request.tags),
        )
        .await
        .map_err(Into::into)
}

pub async fn sync(
    handles: RuntimeHandles,
    request: SyncRequest,
    progress: Option<SurfaceProgressSink>,
) -> anyhow::Result<SyncExecutionSummary> {
    handles
        .sync
        .sync_collection_with_progress(
            request.source,
            request.run_migrations,
            request.run_index,
            request.force_reindex,
            progress,
        )
        .await
        .map_err(Into::into)
}

pub async fn index(
    handles: RuntimeHandles,
    request: IndexRequest,
    progress: Option<SurfaceProgressSink>,
) -> anyhow::Result<IndexExecutionSummary> {
    handles
        .index
        .index_all_notes_with_progress(request.force_reindex, progress)
        .await
        .map_err(Into::into)
}

pub fn generate_preview(
    service: &GeneratePreviewService,
    request: &GenerateRequest,
) -> anyhow::Result<GeneratePreview> {
    service.preview(&request.file).map_err(Into::into)
}

pub fn validate(
    service: &ValidationService,
    request: &ValidateRequest,
) -> anyhow::Result<ValidationSummary> {
    service
        .validate_file(&request.file, request.include_quality)
        .map_err(Into::into)
}

pub fn obsidian_scan(
    service: &ObsidianScanService,
    request: &ObsidianScanRequest,
    progress: Option<SurfaceProgressSink>,
) -> anyhow::Result<ObsidianScanPreview> {
    service
        .scan_with_progress(
            &request.vault,
            &request.source_dirs,
            request.dry_run,
            progress,
        )
        .map_err(Into::into)
}

pub fn tag_audit(
    service: &TagAuditService,
    request: &TagAuditRequest,
) -> anyhow::Result<TagAuditSummary> {
    service
        .audit_file(&request.file, request.apply_fixes)
        .map_err(Into::into)
}

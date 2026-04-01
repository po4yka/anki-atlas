use std::path::PathBuf;

use database::MigrationResult;
use surface_contracts::analytics::{
    DuplicateCluster, DuplicateStats, LabelingStats, TaxonomyLoadSummary, TopicCoverage, TopicGap,
    WeakNote,
};
use surface_contracts::search::{
    ChunkSearchRequest as SurfaceChunkSearchRequest, ChunkSearchResponse,
    SearchFilterInput as SurfaceSearchFilterInput, SearchRequest as SurfaceSearchRequest,
    SearchResponse,
};
use surface_runtime::{
    GeneratePreview, GeneratePreviewService, IndexExecutionSummary, ObsidianScanPreview,
    ObsidianScanService, SurfaceProgressSink, SyncExecutionSummary, TagAuditService,
    TagAuditSummary, ValidationService, ValidationSummary,
};

pub use crate::runtime::RuntimeHandles;

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

pub async fn run_migrations(
    settings: &common::config::Settings,
) -> anyhow::Result<MigrationResult> {
    let pool = database::create_pool(&settings.database()).await?;
    database::run_migrations(&pool).await.map_err(Into::into)
}

pub async fn search(
    handles: RuntimeHandles,
    request: SearchRequest,
) -> anyhow::Result<SearchResponse> {
    let request = SurfaceSearchRequest {
        query: request.query,
        filters: Some(SurfaceSearchFilterInput {
            deck_names: Some(request.deck_names),
            tags: Some(request.tags),
            ..Default::default()
        }),
        limit: request.limit,
        semantic_weight: 1.0,
        fts_weight: 1.0,
        semantic_only: request.semantic_only,
        fts_only: request.fts_only,
        rerank_override: None,
        rerank_top_n_override: None,
    };
    handles.search.search(&request).await.map_err(Into::into)
}

pub async fn search_chunks(
    handles: RuntimeHandles,
    request: ChunkSearchRequest,
) -> anyhow::Result<ChunkSearchResponse> {
    let request = SurfaceChunkSearchRequest {
        query: request.query,
        filters: Some(SurfaceSearchFilterInput {
            deck_names: Some(request.deck_names),
            tags: Some(request.tags),
            ..Default::default()
        }),
        limit: request.limit,
    };
    handles
        .search
        .search_chunks(&request)
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
) -> anyhow::Result<TaxonomyLoadSummary> {
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

use std::path::PathBuf;
use std::sync::Arc;

use common::logging::{LoggingConfig, init_global_logging};
use jobs::types::{IndexJobPayload, SyncJobPayload};
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{Implementation, ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use serde_json::Value;
use surface_contracts::search::{ChunkSearchRequest, SearchFilterInput, SearchRequest};
use surface_runtime::{BuildSurfaceServicesOptions, SurfaceError, SurfaceServices};

use crate::formatters;
use crate::handlers::{error_result, success_result};
use crate::tools::{
    ChunkSearchResultView, ChunkSearchToolInput, ChunkSearchToolResult, DuplicatesToolInput,
    DuplicatesToolResult, GenerateToolInput, IndexJobToolInput, JobAcceptedToolResult,
    JobCancelToolInput, JobStatusToolInput, JobStatusToolResult, ObsidianSyncToolInput,
    SearchResultView, SearchToolInput, SearchToolResult, SyncJobToolInput, TagAuditToolInput,
    ToolError, TopicCoverageToolInput, TopicCoverageToolResult, TopicGapsToolInput,
    TopicGapsToolResult, TopicWeakNotesToolInput, TopicWeakNotesToolResult, TopicsToolInput,
    TopicsToolResult, ValidateToolInput, WorkflowToolResult,
};

#[derive(Clone)]
pub struct AnkiAtlasServer {
    services: Arc<SurfaceServices>,
    tool_router: ToolRouter<Self>,
}

impl AnkiAtlasServer {
    pub fn new(services: Arc<SurfaceServices>) -> Self {
        Self {
            services,
            tool_router: Self::tool_router(),
        }
    }

    pub fn name(&self) -> &str {
        "anki-atlas"
    }

    pub fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }

    pub fn tool_count(&self) -> usize {
        self.tool_router.list_all().len()
    }

    pub fn tool_names(&self) -> Vec<String> {
        self.tool_router
            .list_all()
            .into_iter()
            .map(|tool| tool.name.into_owned())
            .collect()
    }

    fn tool_error(code: &str, message: impl Into<String>, details: Option<String>) -> ToolError {
        ToolError {
            error: code.to_string(),
            message: message.into(),
            details,
        }
    }

    fn surface_error(error: SurfaceError) -> ToolError {
        match error {
            SurfaceError::Unsupported(message) => Self::tool_error("unsupported", message, None),
            SurfaceError::PathNotFound(path) => Self::tool_error(
                "not_found",
                format!("path not found: {}", path.display()),
                None,
            ),
            SurfaceError::NotFound(message) => Self::tool_error("not_found", message, None),
            SurfaceError::InvalidInput(message) => Self::tool_error("invalid_input", message, None),
            SurfaceError::Database(error) => {
                Self::tool_error("database_unavailable", error.to_string(), None)
            }
            SurfaceError::VectorStore(error) => {
                Self::tool_error("vector_store_unavailable", error.to_string(), None)
            }
            SurfaceError::Provider(message) => Self::tool_error("provider_error", message, None),
            other => Self::tool_error("internal_error", other.to_string(), None),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for AnkiAtlasServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::new(ServerCapabilities::builder().enable_tools().build());
        info.server_info = Implementation::new(self.name(), self.version());
        info.instructions = Some(
            "Search and inspect anki-atlas data. Sync/index writes are exposed only as async jobs."
                .to_string(),
        );
        info
    }
}

#[tool_router]
impl AnkiAtlasServer {
    #[tool(
        name = "ankiatlas_search",
        description = "Search notes with the shared hybrid search service"
    )]
    async fn ankiatlas_search(
        &self,
        Parameters(input): Parameters<SearchToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        let request = SearchRequest {
            query: input.query.clone(),
            filters: Some(SearchFilterInput {
                deck_names: Some(input.deck_names.clone()),
                tags: Some(input.tags.clone()),
                ..Default::default()
            }),
            limit: input.limit,
            semantic_weight: 1.0,
            fts_weight: 1.0,
            search_mode: input.search_mode.into(),
            rerank_override: None,
            rerank_top_n_override: None,
        };
        if let Err(error) = request.validate() {
            return error_result(
                input.output_mode,
                Self::tool_error("invalid_input", error, None),
            );
        }
        match self.services.search.search(&request).await {
            Ok(result) => {
                let response = SearchToolResult {
                    query: result.query,
                    total_results: result.results.len(),
                    lexical_mode: format!("{:?}", result.lexical_mode),
                    lexical_fallback_used: result.lexical_fallback_used,
                    rerank_applied: result.rerank_applied,
                    query_suggestions: result.query_suggestions,
                    autocomplete_suggestions: result.autocomplete_suggestions,
                    results: result
                        .results
                        .into_iter()
                        .map(|item| SearchResultView {
                            note_id: item.note_id.into(),
                            rrf_score: item.rrf_score,
                            semantic_score: item.semantic_score,
                            fts_score: item.fts_score,
                            rerank_score: item.rerank_score,
                            headline: item.headline,
                            sources: item.sources,
                            match_modality: item.match_modality,
                            match_chunk_kind: item.match_chunk_kind,
                            match_source_field: item.match_source_field,
                            match_asset_rel_path: item.match_asset_rel_path,
                        })
                        .collect(),
                };
                success_result(
                    input.output_mode,
                    formatters::format_search(&response),
                    &response,
                )
            }
            Err(error) => error_result(
                input.output_mode,
                Self::tool_error("search_error", error.to_string(), None),
            ),
        }
    }

    #[tool(
        name = "ankiatlas_search_chunks",
        description = "Search raw multimodal chunks with semantic retrieval only"
    )]
    async fn ankiatlas_search_chunks(
        &self,
        Parameters(input): Parameters<ChunkSearchToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        let request = ChunkSearchRequest {
            query: input.query.clone(),
            filters: Some(SearchFilterInput {
                deck_names: Some(input.deck_names.clone()),
                tags: Some(input.tags.clone()),
                ..Default::default()
            }),
            limit: input.limit,
        };
        if let Err(error) = request.validate() {
            return error_result(
                input.output_mode,
                Self::tool_error("invalid_input", error, None),
            );
        }
        match self.services.search.search_chunks(&request).await {
            Ok(result) => {
                let response = ChunkSearchToolResult {
                    query: result.query,
                    total_results: result.results.len(),
                    results: result
                        .results
                        .into_iter()
                        .map(|item| ChunkSearchResultView {
                            note_id: item.note_id.into(),
                            chunk_id: item.chunk_id,
                            chunk_kind: item.chunk_kind,
                            modality: item.modality,
                            source_field: item.source_field,
                            asset_rel_path: item.asset_rel_path,
                            mime_type: item.mime_type,
                            preview_label: item.preview_label,
                            score: item.score,
                        })
                        .collect(),
                };
                success_result(
                    input.output_mode,
                    formatters::format_chunk_search(&response),
                    &response,
                )
            }
            Err(error) => error_result(
                input.output_mode,
                Self::tool_error("search_error", error.to_string(), None),
            ),
        }
    }

    #[tool(name = "ankiatlas_topics", description = "Inspect the taxonomy tree")]
    async fn ankiatlas_topics(
        &self,
        Parameters(input): Parameters<TopicsToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        match self
            .services
            .analytics
            .get_taxonomy_tree(input.root_path.clone())
            .await
        {
            Ok(topics) => {
                let response = TopicsToolResult {
                    root_path: input.root_path,
                    topic_count: topics.len(),
                    topics: serde_json::json!(topics),
                };
                success_result(
                    input.output_mode,
                    formatters::format_topics(&response),
                    &response,
                )
            }
            Err(error) => error_result(
                input.output_mode,
                Self::tool_error("analytics_error", error.to_string(), None),
            ),
        }
    }

    #[tool(
        name = "ankiatlas_topic_coverage",
        description = "Inspect topic coverage metrics"
    )]
    async fn ankiatlas_topic_coverage(
        &self,
        Parameters(input): Parameters<TopicCoverageToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        match self
            .services
            .analytics
            .get_coverage(input.topic_path.clone(), input.include_subtree)
            .await
        {
            Ok(coverage) => {
                let response = TopicCoverageToolResult {
                    topic_path: input.topic_path,
                    found: coverage.is_some(),
                    coverage: coverage
                        .map(|value| serde_json::to_value(value).unwrap_or(Value::Null)),
                };
                success_result(
                    input.output_mode,
                    formatters::format_coverage(&response),
                    &response,
                )
            }
            Err(error) => error_result(
                input.output_mode,
                Self::tool_error("analytics_error", error.to_string(), None),
            ),
        }
    }

    #[tool(
        name = "ankiatlas_topic_gaps",
        description = "Inspect topic gap candidates"
    )]
    async fn ankiatlas_topic_gaps(
        &self,
        Parameters(input): Parameters<TopicGapsToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        match self
            .services
            .analytics
            .get_gaps(input.topic_path.clone(), input.min_coverage)
            .await
        {
            Ok(gaps) => {
                let response = TopicGapsToolResult {
                    topic_path: input.topic_path,
                    min_coverage: input.min_coverage,
                    gaps: serde_json::to_value(gaps).unwrap_or(Value::Null),
                };
                success_result(
                    input.output_mode,
                    formatters::format_gaps(&response),
                    &response,
                )
            }
            Err(error) => error_result(
                input.output_mode,
                Self::tool_error("analytics_error", error.to_string(), None),
            ),
        }
    }

    #[tool(
        name = "ankiatlas_topic_weak_notes",
        description = "List weak notes for a topic"
    )]
    async fn ankiatlas_topic_weak_notes(
        &self,
        Parameters(input): Parameters<TopicWeakNotesToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        match self
            .services
            .analytics
            .get_weak_notes(input.topic_path.clone(), input.max_results)
            .await
        {
            Ok(notes) => {
                let response = TopicWeakNotesToolResult {
                    topic_path: input.topic_path,
                    max_results: input.max_results,
                    notes: serde_json::to_value(notes).unwrap_or(Value::Null),
                };
                success_result(
                    input.output_mode,
                    formatters::format_weak_notes(&response),
                    &response,
                )
            }
            Err(error) => error_result(
                input.output_mode,
                Self::tool_error("analytics_error", error.to_string(), None),
            ),
        }
    }

    #[tool(
        name = "ankiatlas_duplicates",
        description = "Find duplicate-note clusters"
    )]
    async fn ankiatlas_duplicates(
        &self,
        Parameters(input): Parameters<DuplicatesToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        match self
            .services
            .analytics
            .find_duplicates(
                input.threshold,
                input.max_clusters,
                (!input.deck_filter.is_empty()).then(|| input.deck_filter.clone()),
                (!input.tag_filter.is_empty()).then(|| input.tag_filter.clone()),
            )
            .await
        {
            Ok((clusters, stats)) => {
                let response = DuplicatesToolResult {
                    threshold: input.threshold,
                    max_clusters: input.max_clusters,
                    clusters: serde_json::to_value(clusters).unwrap_or(Value::Null),
                    stats: serde_json::to_value(stats).unwrap_or(Value::Null),
                };
                success_result(
                    input.output_mode,
                    formatters::format_duplicates(&response),
                    &response,
                )
            }
            Err(error) => error_result(
                input.output_mode,
                Self::tool_error("analytics_error", error.to_string(), None),
            ),
        }
    }

    #[tool(name = "ankiatlas_sync_job", description = "Enqueue a sync job")]
    async fn ankiatlas_sync_job(
        &self,
        Parameters(input): Parameters<SyncJobToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        match self
            .services
            .job_manager
            .enqueue_sync_job(
                SyncJobPayload {
                    source: input.source,
                    run_migrations: input.run_migrations,
                    index: input.index,
                    force_reindex: input.force_reindex,
                },
                None,
            )
            .await
        {
            Ok(record) => {
                let response = JobAcceptedToolResult {
                    job_id: record.job_id.clone(),
                    job_type: record.job_type.to_string(),
                    status: record.status.to_string(),
                    poll_hint: format!("call ankiatlas_job_status with job_id={}", record.job_id),
                    cancel_hint: format!("call ankiatlas_job_cancel with job_id={}", record.job_id),
                };
                success_result(
                    input.output_mode,
                    formatters::format_job_accepted(&response),
                    &response,
                )
            }
            Err(error) => error_result(
                input.output_mode,
                Self::tool_error("job_error", error.to_string(), None),
            ),
        }
    }

    #[tool(name = "ankiatlas_index_job", description = "Enqueue an index job")]
    async fn ankiatlas_index_job(
        &self,
        Parameters(input): Parameters<IndexJobToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        match self
            .services
            .job_manager
            .enqueue_index_job(
                IndexJobPayload {
                    force_reindex: input.force_reindex,
                },
                None,
            )
            .await
        {
            Ok(record) => {
                let response = JobAcceptedToolResult {
                    job_id: record.job_id.clone(),
                    job_type: record.job_type.to_string(),
                    status: record.status.to_string(),
                    poll_hint: format!("call ankiatlas_job_status with job_id={}", record.job_id),
                    cancel_hint: format!("call ankiatlas_job_cancel with job_id={}", record.job_id),
                };
                success_result(
                    input.output_mode,
                    formatters::format_job_accepted(&response),
                    &response,
                )
            }
            Err(error) => error_result(
                input.output_mode,
                Self::tool_error("job_error", error.to_string(), None),
            ),
        }
    }

    #[tool(name = "ankiatlas_job_status", description = "Inspect a queued job")]
    async fn ankiatlas_job_status(
        &self,
        Parameters(input): Parameters<JobStatusToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        match self.services.job_manager.get_job(&input.job_id).await {
            Ok(record) => {
                let response = JobStatusToolResult {
                    job_id: record.job_id,
                    job_type: record.job_type.to_string(),
                    status: record.status.to_string(),
                    progress: record.progress,
                    message: record.message,
                    result: record
                        .result
                        .map(|value| serde_json::to_value(value).unwrap_or(Value::Null)),
                    error: record.error,
                };
                success_result(
                    input.output_mode,
                    formatters::format_job_status(&response),
                    &response,
                )
            }
            Err(error) => error_result(
                input.output_mode,
                Self::tool_error("job_error", error.to_string(), None),
            ),
        }
    }

    #[tool(
        name = "ankiatlas_job_cancel",
        description = "Request job cancellation"
    )]
    async fn ankiatlas_job_cancel(
        &self,
        Parameters(input): Parameters<JobCancelToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        match self.services.job_manager.cancel_job(&input.job_id).await {
            Ok(record) => {
                let response = JobStatusToolResult {
                    job_id: record.job_id,
                    job_type: record.job_type.to_string(),
                    status: record.status.to_string(),
                    progress: record.progress,
                    message: record.message,
                    result: record
                        .result
                        .map(|value| serde_json::to_value(value).unwrap_or(Value::Null)),
                    error: record.error,
                };
                success_result(
                    input.output_mode,
                    formatters::format_job_status(&response),
                    &response,
                )
            }
            Err(error) => error_result(
                input.output_mode,
                Self::tool_error("job_error", error.to_string(), None),
            ),
        }
    }

    #[tool(
        name = "ankiatlas_generate",
        description = "Preview note generation from a markdown file"
    )]
    async fn ankiatlas_generate(
        &self,
        Parameters(input): Parameters<GenerateToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        match self
            .services
            .generate_preview
            .preview(PathBuf::from(&input.file_path).as_path())
        {
            Ok(preview) => {
                let response = WorkflowToolResult {
                    path: input.file_path,
                    summary: format!("estimated cards: {}", preview.estimated_cards),
                    data: serde_json::to_value(preview).unwrap_or(Value::Null),
                };
                success_result(
                    input.output_mode,
                    formatters::format_workflow(&response),
                    &response,
                )
            }
            Err(error) => error_result(input.output_mode, Self::surface_error(error)),
        }
    }

    #[tool(
        name = "ankiatlas_validate",
        description = "Validate card content from a file"
    )]
    async fn ankiatlas_validate(
        &self,
        Parameters(input): Parameters<ValidateToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        match self
            .services
            .validation
            .validate_file(PathBuf::from(&input.file_path).as_path(), input.quality)
        {
            Ok(summary) => {
                let response = WorkflowToolResult {
                    path: input.file_path,
                    summary: format!("valid={} issues={}", summary.is_valid, summary.issues.len()),
                    data: serde_json::to_value(summary).unwrap_or(Value::Null),
                };
                success_result(
                    input.output_mode,
                    formatters::format_workflow(&response),
                    &response,
                )
            }
            Err(error) => error_result(input.output_mode, Self::surface_error(error)),
        }
    }

    #[tool(
        name = "ankiatlas_obsidian_sync",
        description = "Preview an Obsidian vault scan"
    )]
    async fn ankiatlas_obsidian_sync(
        &self,
        Parameters(input): Parameters<ObsidianSyncToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        match self.services.obsidian_scan.scan(
            PathBuf::from(&input.vault_path).as_path(),
            &input.source_dirs,
            input.dry_run,
        ) {
            Ok(summary) => {
                let response = WorkflowToolResult {
                    path: input.vault_path,
                    summary: format!(
                        "notes={} generated_cards={}",
                        summary.note_count, summary.generated_cards
                    ),
                    data: serde_json::to_value(summary).unwrap_or(Value::Null),
                };
                success_result(
                    input.output_mode,
                    formatters::format_workflow(&response),
                    &response,
                )
            }
            Err(error) => error_result(input.output_mode, Self::surface_error(error)),
        }
    }

    #[tool(
        name = "ankiatlas_tag_audit",
        description = "Validate and normalize tag files"
    )]
    async fn ankiatlas_tag_audit(
        &self,
        Parameters(input): Parameters<TagAuditToolInput>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        match self
            .services
            .tag_audit
            .audit_file(PathBuf::from(&input.file_path).as_path(), input.fix)
        {
            Ok(summary) => {
                let response = WorkflowToolResult {
                    path: input.file_path,
                    summary: format!(
                        "entries={} applied_fixes={}",
                        summary.entries.len(),
                        summary.applied_fixes
                    ),
                    data: serde_json::to_value(summary).unwrap_or(Value::Null),
                };
                success_result(
                    input.output_mode,
                    formatters::format_workflow(&response),
                    &response,
                )
            }
            Err(error) => error_result(input.output_mode, Self::surface_error(error)),
        }
    }
}

pub async fn run_server() -> anyhow::Result<()> {
    let _ = init_global_logging(&LoggingConfig {
        debug: false,
        json_output: true,
    });

    let settings = common::config::Settings::load()?;
    let services = Arc::new(
        surface_runtime::build_surface_services(
            &settings,
            BuildSurfaceServicesOptions {
                enable_direct_execution: false,
            },
        )
        .await?,
    );

    let transport = rmcp::transport::stdio();
    let server = AnkiAtlasServer::new(services).serve(transport).await?;
    server.waiting().await?;
    Ok(())
}

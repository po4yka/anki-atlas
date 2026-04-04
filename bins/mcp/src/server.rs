use std::path::PathBuf;
use std::sync::Arc;

use common::logging::{LoggingConfig, init_global_logging};
use jobs::types::{IndexJobPayload, SyncJobPayload};
use rmcp::handler::server::{router::tool::ToolRouter, wrapper::Parameters};
use rmcp::model::{
    Annotated, GetPromptRequestParams, GetPromptResult, Implementation, ListPromptsResult,
    ListResourceTemplatesResult, ListResourcesResult, PaginatedRequestParams, Prompt,
    PromptArgument, PromptMessage, PromptMessageRole, RawResource, RawResourceTemplate,
    ReadResourceRequestParams, ReadResourceResult, ResourceContents, ServerCapabilities,
    ServerInfo,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData, RoleServer, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
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

#[allow(clippy::field_reassign_with_default, clippy::manual_async_fn)]
#[tool_handler(router = self.tool_router)]
impl ServerHandler for AnkiAtlasServer {
    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_prompts()
                .enable_resources()
                .build(),
        );
        info.server_info = Implementation::new(self.name(), self.version());
        info.instructions = Some(
            "Search and inspect anki-atlas data. Browse resources for taxonomy and stats. \
             Use prompts for common workflows. Sync/index writes are exposed only as async jobs."
                .to_string(),
        );
        info
    }

    fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListResourcesResult, ErrorData>> + Send + '_ {
        let mut result = ListResourcesResult::default();
        result.resources = vec![
            Annotated::new(
                RawResource::new("anki://taxonomy", "Topic Taxonomy")
                    .with_description("Full topic taxonomy tree as JSON")
                    .with_mime_type("application/json"),
                None,
            ),
            Annotated::new(
                RawResource::new("anki://stats", "Collection Stats")
                    .with_description("Card counts, coverage summary, and index status")
                    .with_mime_type("application/json"),
                None,
            ),
        ];
        std::future::ready(Ok(result))
    }

    fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListResourceTemplatesResult, ErrorData>> + Send + '_
    {
        let mut result = ListResourceTemplatesResult::default();
        result.resource_templates = vec![Annotated::new(
            RawResourceTemplate::new("anki://taxonomy/{path}", "Topic Coverage")
                .with_description("Coverage metrics for a specific topic path"),
            None,
        )];
        std::future::ready(Ok(result))
    }

    fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ReadResourceResult, ErrorData>> + Send + '_ {
        async move {
            let uri = request.uri.as_str();
            let make_result = |json: String, u: &str| {
                ReadResourceResult::new(vec![
                    ResourceContents::text(json, u).with_mime_type("application/json"),
                ])
            };
            match uri {
                "anki://taxonomy" => {
                    let tree = self
                        .services
                        .analytics
                        .get_taxonomy_tree(None)
                        .await
                        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                    let json = serde_json::to_string_pretty(&tree)
                        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                    Ok(make_result(json, uri))
                }
                "anki://stats" => {
                    let stats = serde_json::json!({
                        "note_count": sqlx::query_scalar::<_, i64>(
                            "SELECT COUNT(*) FROM notes WHERE deleted_at IS NULL"
                        )
                        .fetch_one(&self.services.db)
                        .await
                        .unwrap_or(0),
                        "card_count": sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM cards")
                            .fetch_one(&self.services.db)
                            .await
                            .unwrap_or(0),
                        "topic_count": sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM topics")
                            .fetch_one(&self.services.db)
                            .await
                            .unwrap_or(0),
                    });
                    let json = serde_json::to_string_pretty(&stats)
                        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                    Ok(make_result(json, uri))
                }
                _ if uri.starts_with("anki://taxonomy/") => {
                    let path = uri
                        .strip_prefix("anki://taxonomy/")
                        .unwrap_or("")
                        .to_string();
                    let coverage = self
                        .services
                        .analytics
                        .get_coverage(path.clone(), true)
                        .await
                        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                    let json = serde_json::to_string_pretty(&coverage)
                        .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                    Ok(make_result(json, uri))
                }
                _ => Err(ErrorData::resource_not_found(
                    "resource_not_found",
                    Some(serde_json::json!({"uri": uri})),
                )),
            }
        }
    }

    fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListPromptsResult, ErrorData>> + Send + '_ {
        fn prompt_arg(name: &str, desc: &str, required: bool) -> PromptArgument {
            let mut arg = PromptArgument::new(name);
            arg.description = Some(desc.to_string());
            arg.required = Some(required);
            arg
        }

        let mut result = ListPromptsResult::default();
        result.prompts = vec![
            Prompt::new(
                "generate_cards",
                Some("Generate Anki flashcards from a topic"),
                Some(vec![
                    prompt_arg("topic", "Topic to generate cards for", true),
                    prompt_arg("count", "Number of cards (default: 5)", false),
                ]),
            ),
            Prompt::new(
                "find_gaps",
                Some("Find knowledge coverage gaps in a topic area"),
                Some(vec![prompt_arg("topic", "Topic path to analyze", true)]),
            ),
            Prompt::new(
                "review_card",
                Some("Review and improve an existing card"),
                Some(vec![prompt_arg(
                    "query",
                    "Card content or search query",
                    true,
                )]),
            ),
            Prompt::new(
                "explain_topic",
                Some("Explain a topic with related concepts"),
                Some(vec![prompt_arg("topic", "Topic to explain", true)]),
            ),
        ];
        std::future::ready(Ok(result))
    }

    fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<GetPromptResult, ErrorData>> + Send + '_ {
        async move {
            let args = request.arguments.unwrap_or_default();
            let arg_str = |key: &str, default: &str| -> String {
                args.get(key)
                    .and_then(|v| v.as_str())
                    .unwrap_or(default)
                    .to_string()
            };
            match request.name.as_str() {
                "generate_cards" => {
                    let topic = arg_str("topic", "general");
                    let count = args
                        .get("count")
                        .and_then(|v| v.as_str())
                        .and_then(|c| c.parse::<usize>().ok())
                        .unwrap_or(5);
                    Ok(GetPromptResult::new(vec![PromptMessage::new_text(
                        PromptMessageRole::User,
                        format!(
                            "Generate {count} Anki flashcards about '{topic}'. \
                             Include a mix of card types: basic Q&A, cloze, and MCQ. \
                             Test understanding and reasoning, not rote memorization. \
                             Format as JSON array: card_type, front, back, tags. \
                             Cards must be bilingual (EN + RU)."
                        ),
                    )])
                    .with_description("Generate Anki flashcards"))
                }
                "find_gaps" => {
                    let topic = arg_str("topic", "programming");
                    Ok(GetPromptResult::new(vec![PromptMessage::new_text(
                        PromptMessageRole::User,
                        format!(
                            "Analyze '{topic}' for knowledge gaps. \
                             Use ankiatlas_topic_gaps to find missing subtopics. \
                             For each gap, suggest flashcard topics to fill it. \
                             Prioritize by importance."
                        ),
                    )])
                    .with_description("Find knowledge gaps"))
                }
                "review_card" => {
                    let query = arg_str("query", "");
                    Ok(GetPromptResult::new(vec![PromptMessage::new_text(
                        PromptMessageRole::User,
                        format!(
                            "Find and review the card matching '{query}'. \
                             Use ankiatlas_search to locate it. Evaluate: \
                             1) Atomic? 2) Tests reasoning? 3) Clear wording? \
                             4) Correct tags? Suggest improvements."
                        ),
                    )])
                    .with_description("Review a card"))
                }
                "explain_topic" => {
                    let topic = arg_str("topic", "");
                    Ok(GetPromptResult::new(vec![PromptMessage::new_text(
                        PromptMessageRole::User,
                        format!(
                            "Explain '{topic}' using the anki-atlas knowledge base. \
                             Use ankiatlas_search for existing cards, \
                             ankiatlas_topic_coverage for status. \
                             Present: key concepts, prerequisites, related topics, gaps."
                        ),
                    )])
                    .with_description("Explain a topic"))
                }
                _ => Err(ErrorData::invalid_params(
                    format!("unknown prompt: {}", request.name),
                    None,
                )),
            }
        }
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
                    reindex_mode: input.reindex_mode.into(),
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
                    reindex_mode: input.reindex_mode.into(),
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
        match self.services.validation.validate_file(
            PathBuf::from(&input.file_path).as_path(),
            if input.quality {
                surface_runtime::QualityCheck::Include
            } else {
                surface_runtime::QualityCheck::Skip
            },
        ) {
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
            if input.dry_run {
                common::ExecutionMode::DryRun
            } else {
                common::ExecutionMode::Execute
            },
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

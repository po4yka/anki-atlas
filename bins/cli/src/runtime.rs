use std::sync::Arc;

use common::config::{EmbeddingProviderKind, Settings};
use surface_runtime::{
    AnalyticsFacade, BuildSurfaceServicesOptions, GeneratePreviewService, IndexExecutor,
    ObsidianScanService, SearchFacade, SurfaceServices, SyncExecutionHandle, TagAuditService,
    ValidationService, build_surface_services,
};

use crate::args::Commands;
use crate::commands;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    ReadOnly,
    DirectExecution,
}

impl ExecutionMode {
    fn enable_direct_execution(self) -> bool {
        matches!(self, Self::DirectExecution)
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeSettingsSummary {
    pub postgres_url: String,
    pub qdrant_url: String,
    pub job_backend: String,
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

fn embedding_provider_label(provider: EmbeddingProviderKind) -> &'static str {
    match provider {
        EmbeddingProviderKind::OpenAi => "openai",
        EmbeddingProviderKind::Google => "google",
        EmbeddingProviderKind::Mock => "mock",
        EmbeddingProviderKind::FastEmbed => "fastembed",
    }
}

pub fn summarize_settings(settings: &Settings) -> RuntimeSettingsSummary {
    RuntimeSettingsSummary {
        postgres_url: settings.postgres_url.clone(),
        qdrant_url: settings.qdrant_url.clone(),
        job_backend: "postgresql".to_string(),
        embedding_provider: embedding_provider_label(settings.embedding_provider).to_string(),
        embedding_model: settings.embedding_model.clone(),
        rerank_enabled: settings.rerank_enabled,
    }
}

pub async fn bootstrap_runtime(mode: ExecutionMode) -> anyhow::Result<RuntimeBootstrap> {
    let settings = Settings::load()?;
    let services = build_surface_services(
        &settings,
        BuildSurfaceServicesOptions {
            enable_direct_execution: mode.enable_direct_execution(),
        },
    )
    .await?;

    Ok(RuntimeBootstrap {
        summary: summarize_settings(&settings),
        settings,
        services: Arc::new(services),
    })
}

fn execution_mode_for(command: &Commands) -> Option<ExecutionMode> {
    match command {
        Commands::Sync(_) | Commands::Index(_) => Some(ExecutionMode::DirectExecution),
        Commands::Search(_)
        | Commands::Topics(_)
        | Commands::Coverage(_)
        | Commands::Gaps(_)
        | Commands::WeakNotes(_)
        | Commands::Duplicates(_) => Some(ExecutionMode::ReadOnly),
        Commands::Version
        | Commands::Migrate
        | Commands::Tui
        | Commands::Generate(_)
        | Commands::Validate(_)
        | Commands::ObsidianSync(_)
        | Commands::TagAudit(_)
        | Commands::Cardloop(_) => None,
    }
}

pub async fn dispatch_service_command(command: &Commands) -> anyhow::Result<()> {
    let mode = execution_mode_for(command)
        .ok_or_else(|| anyhow::anyhow!("command does not use the shared runtime"))?;
    let runtime = bootstrap_runtime(mode).await?;

    match command {
        Commands::Sync(args) => commands::sync::run(args, &runtime.services).await,
        Commands::Index(args) => commands::index::run(args, &runtime.services).await,
        Commands::Search(args) => commands::search::run(args, &runtime.services).await,
        Commands::Topics(args) => commands::topics::run(args, &runtime.services).await,
        Commands::Coverage(args) => commands::coverage::run(args, &runtime.services).await,
        Commands::Gaps(args) => commands::gaps::run(args, &runtime.services).await,
        Commands::WeakNotes(args) => commands::weak_notes::run(args, &runtime.services).await,
        Commands::Duplicates(args) => commands::duplicates::run(args, &runtime.services).await,
        _ => unreachable!("execution mode is only defined for service-backed commands"),
    }
}

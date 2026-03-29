use std::collections::VecDeque;
use std::sync::Arc;

use common::config::Settings;
use surface_contracts::analytics::{
    DuplicateCluster, DuplicateStats, TopicCoverage, TopicGap, WeakNote,
};
use surface_contracts::search::SearchResponse;
use surface_runtime::{
    GeneratePreview, IndexExecutionSummary, ObsidianScanPreview, SurfaceProgressEvent,
    SurfaceServices, SyncExecutionSummary, TagAuditSummary, ValidationSummary,
};

use crate::runtime::{RuntimeBootstrap, RuntimeHandles, RuntimeSettingsSummary};

use super::{SearchState, TopicsState, WorkflowsState};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Screen {
    Home,
    Search,
    Topics,
    Workflows,
}

impl Screen {
    pub(crate) const ALL: [Screen; 4] = [
        Screen::Home,
        Screen::Search,
        Screen::Topics,
        Screen::Workflows,
    ];

    pub(crate) fn title(&self) -> &'static str {
        match self {
            Self::Home => "Home",
            Self::Search => "Search",
            Self::Topics => "Topics",
            Self::Workflows => "Workflows",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FocusRegion {
    Navigation,
    Content,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModalState {
    Help,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RuntimeStatus {
    Bootstrapping,
    Ready,
    Error,
}

#[derive(Clone)]
pub(crate) struct RuntimeState {
    pub(crate) status: RuntimeStatus,
    pub(crate) summary: Option<RuntimeSettingsSummary>,
    pub(crate) services: Option<Arc<SurfaceServices>>,
    pub(crate) settings: Option<Settings>,
    pub(crate) error: Option<String>,
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            status: RuntimeStatus::Bootstrapping,
            summary: None,
            services: None,
            settings: None,
            error: None,
        }
    }
}

#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) screen: Screen,
    pub(crate) navigation_index: usize,
    pub(crate) focus_region: FocusRegion,
    pub(crate) modal: Option<ModalState>,
    pub(crate) runtime: RuntimeState,
    pub(crate) activity: VecDeque<String>,
    pub(crate) busy_label: Option<String>,
    pub(crate) progress: Option<SurfaceProgressEvent>,
    pub(crate) last_result_summary: Option<String>,
    pub(crate) search: SearchState,
    pub(crate) topics: TopicsState,
    pub(crate) workflows: WorkflowsState,
    pub(crate) should_quit: bool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            screen: Screen::Home,
            navigation_index: 0,
            focus_region: FocusRegion::Navigation,
            modal: None,
            runtime: RuntimeState::default(),
            activity: VecDeque::new(),
            busy_label: Some("runtime bootstrap".to_string()),
            progress: None,
            last_result_summary: None,
            search: SearchState::default(),
            topics: TopicsState::default(),
            workflows: WorkflowsState::default(),
            should_quit: false,
        }
    }
}

impl AppState {
    pub(crate) fn push_activity(&mut self, message: String) {
        self.activity.push_front(message);
        while self.activity.len() > 8 {
            let _ = self.activity.pop_back();
        }
    }

    pub(crate) fn current_screen(&self) -> Screen {
        Screen::ALL[self.navigation_index]
    }

    pub(crate) fn set_screen(&mut self, screen: Screen) {
        self.screen = screen;
        self.navigation_index = Screen::ALL
            .iter()
            .position(|candidate| *candidate == screen)
            .unwrap_or(0);
    }

    pub(crate) fn runtime_handles(&self) -> Option<RuntimeHandles> {
        self.runtime.services.as_deref().map(RuntimeHandles::from)
    }

    pub(crate) fn apply_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::TaskFinished { label, result } => {
                self.busy_label = None;
                self.progress = None;
                match *result {
                    Ok(TaskResult::Bootstrap(runtime)) => {
                        self.runtime.status = RuntimeStatus::Ready;
                        self.runtime.summary = Some(runtime.summary);
                        self.runtime.services = Some(runtime.services);
                        self.runtime.settings = Some(runtime.settings);
                        self.runtime.error = None;
                        self.last_result_summary = Some("runtime ready".to_string());
                        self.push_activity("runtime bootstrap succeeded".to_string());
                    }
                    Ok(TaskResult::Search(result)) => {
                        self.search.result = Some(result.clone());
                        self.search.error = None;
                        self.search.selected_result = 0;
                        self.last_result_summary =
                            Some(format!("search returned {} results", result.results.len()));
                        self.push_activity("search finished".to_string());
                    }
                    Ok(TaskResult::TopicsTree(result)) => {
                        self.topics.tree_result = Some(result.clone());
                        self.topics.error = None;
                        self.last_result_summary =
                            Some(format!("loaded {} taxonomy nodes", result.len()));
                        self.push_activity("topics tree loaded".to_string());
                    }
                    Ok(TaskResult::Coverage(result)) => {
                        self.topics.coverage_result = Some(result.clone());
                        self.topics.error = None;
                        self.last_result_summary = Some(format!("coverage topic {}", result.path));
                        self.push_activity("coverage lookup finished".to_string());
                    }
                    Ok(TaskResult::Gaps(result)) => {
                        self.topics.gaps_result = Some(result.clone());
                        self.topics.error = None;
                        self.last_result_summary = Some(format!("{} gaps found", result.len()));
                        self.push_activity("gap analysis finished".to_string());
                    }
                    Ok(TaskResult::WeakNotes(result)) => {
                        self.topics.weak_notes_result = Some(result.clone());
                        self.topics.error = None;
                        self.last_result_summary =
                            Some(format!("{} weak notes found", result.len()));
                        self.push_activity("weak notes analysis finished".to_string());
                    }
                    Ok(TaskResult::Duplicates(clusters, stats)) => {
                        self.topics.duplicates_result = Some((clusters, stats.clone()));
                        self.topics.error = None;
                        self.last_result_summary =
                            Some(format!("{} duplicate clusters found", stats.clusters_found));
                        self.push_activity("duplicate analysis finished".to_string());
                    }
                    Ok(TaskResult::Sync(result)) => {
                        self.workflows.sync_result = Some(result.clone());
                        self.workflows.error = None;
                        self.last_result_summary =
                            Some(format!("sync upserted {} notes", result.sync.notes_upserted));
                        self.push_activity("sync finished".to_string());
                    }
                    Ok(TaskResult::Index(result)) => {
                        self.workflows.index_result = Some(result.clone());
                        self.workflows.error = None;
                        self.last_result_summary = Some(format!(
                            "index embedded {} notes",
                            result.stats.notes_embedded
                        ));
                        self.push_activity("index finished".to_string());
                    }
                    Ok(TaskResult::Generate(result)) => {
                        self.workflows.generate_result = Some(result.clone());
                        self.workflows.error = None;
                        self.last_result_summary =
                            Some(format!("previewed {} cards", result.estimated_cards));
                        self.push_activity("generate preview finished".to_string());
                    }
                    Ok(TaskResult::Validate(result)) => {
                        self.workflows.validate_result = Some(result.clone());
                        self.workflows.error = None;
                        self.last_result_summary =
                            Some(format!("validation issues: {}", result.issues.len()));
                        self.push_activity("validation finished".to_string());
                    }
                    Ok(TaskResult::Obsidian(result)) => {
                        self.workflows.obsidian_result = Some(result.clone());
                        self.workflows.error = None;
                        self.last_result_summary =
                            Some(format!("obsidian notes scanned: {}", result.note_count));
                        self.push_activity("obsidian scan finished".to_string());
                    }
                    Ok(TaskResult::TagAudit(result)) => {
                        self.workflows.tag_audit_result = Some(result.clone());
                        self.workflows.error = None;
                        self.last_result_summary =
                            Some(format!("audited {} tags", result.entries.len()));
                        self.push_activity("tag audit finished".to_string());
                    }
                    Err(error) => {
                        if label == "runtime bootstrap" {
                            self.runtime.status = RuntimeStatus::Error;
                            self.runtime.error = Some(error.clone());
                            self.runtime.services = None;
                            self.runtime.summary = None;
                            self.runtime.settings = None;
                        } else {
                            match self.screen {
                                Screen::Search => self.search.error = Some(error.clone()),
                                Screen::Topics => self.topics.error = Some(error.clone()),
                                Screen::Workflows => self.workflows.error = Some(error.clone()),
                                Screen::Home => {}
                            }
                        }
                        self.last_result_summary = Some(format!("{label} failed"));
                        self.push_activity(format!("{label} failed: {error}"));
                    }
                }
            }
            AppEvent::Progress(progress) => {
                self.progress = Some(progress);
            }
            AppEvent::Quit => {
                self.should_quit = true;
            }
        }
    }
}

pub(crate) enum TaskResult {
    Bootstrap(RuntimeBootstrap),
    Search(SearchResponse),
    TopicsTree(Vec<serde_json::Value>),
    Coverage(TopicCoverage),
    Gaps(Vec<TopicGap>),
    WeakNotes(Vec<WeakNote>),
    Duplicates(Vec<DuplicateCluster>, DuplicateStats),
    Sync(SyncExecutionSummary),
    Index(IndexExecutionSummary),
    Generate(GeneratePreview),
    Validate(ValidationSummary),
    Obsidian(ObsidianScanPreview),
    TagAudit(TagAuditSummary),
}

pub(crate) enum AppEvent {
    TaskFinished {
        label: String,
        result: Box<Result<TaskResult, String>>,
    },
    Progress(SurfaceProgressEvent),
    Quit,
}

use std::collections::VecDeque;
use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::Duration;

use analytics::coverage::{TopicCoverage, TopicGap, WeakNote};
use analytics::duplicates::{DuplicateCluster, DuplicateStats};
use common::config::Settings;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Frame;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Tabs, Wrap};
use search::service::HybridSearchResult;
use surface_runtime::{
    GeneratePreview, IndexExecutionSummary, ObsidianScanPreview, SurfaceProgressEvent,
    SurfaceProgressSink, SurfaceServices, SyncExecutionSummary, TagAuditSummary, ValidationSummary,
};
use tokio::sync::mpsc;

use crate::usecases::{
    self, CoverageRequest, DuplicatesRequest, GapsRequest, GenerateRequest, IndexRequest,
    ObsidianScanRequest, RuntimeBootstrap, RuntimeHandles, RuntimeSettingsSummary, SearchRequest,
    SyncRequest, TagAuditRequest, TopicsTreeRequest, ValidateRequest, WeakNotesRequest,
};

pub async fn run() -> anyhow::Result<()> {
    install_panic_hook();
    let mut terminal = TerminalGuard::enter()?;
    let (tx, mut rx) = mpsc::unbounded_channel();
    spawn_ctrl_c_listener(tx.clone());

    let mut app = AppState::default();
    app.push_activity("bootstrapping runtime".to_string());
    start_bootstrap(tx.clone(), true);

    while !app.should_quit {
        terminal.draw(|frame| render(frame, &app))?;

        while let Ok(event) = rx.try_recv() {
            app.apply_event(event);
        }

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                handle_key_event(&mut app, key, &tx);
            }
        }
    }

    Ok(())
}

fn install_panic_hook() {
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen);
        original_hook(info);
    }));
}

fn spawn_ctrl_c_listener(tx: mpsc::UnboundedSender<AppEvent>) {
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            let _ = tx.send(AppEvent::Quit);
        }
    });
}

fn start_bootstrap(tx: mpsc::UnboundedSender<AppEvent>, enable_direct_execution: bool) {
    tokio::spawn(async move {
        let event = match usecases::bootstrap_runtime(enable_direct_execution).await {
            Ok(runtime) => AppEvent::TaskFinished {
                label: "runtime bootstrap".to_string(),
                result: Box::new(Ok(TaskResult::Bootstrap(runtime))),
            },
            Err(error) => AppEvent::TaskFinished {
                label: "runtime bootstrap".to_string(),
                result: Box::new(Err(error.to_string())),
            },
        };
        let _ = tx.send(event);
    });
}

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    fn enter() -> anyhow::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }

    fn draw<F>(&mut self, draw_fn: F) -> anyhow::Result<()>
    where
        F: FnOnce(&mut Frame<'_>),
    {
        self.terminal.draw(draw_fn)?;
        Ok(())
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    Home,
    Search,
    Topics,
    Workflows,
}

impl Screen {
    const ALL: [Screen; 4] = [
        Screen::Home,
        Screen::Search,
        Screen::Topics,
        Screen::Workflows,
    ];

    fn title(&self) -> &'static str {
        match self {
            Self::Home => "Home",
            Self::Search => "Search",
            Self::Topics => "Topics",
            Self::Workflows => "Workflows",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FocusRegion {
    Navigation,
    Content,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModalState {
    Help,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeStatus {
    Bootstrapping,
    Ready,
    Error,
}

#[derive(Clone)]
struct RuntimeState {
    status: RuntimeStatus,
    summary: Option<RuntimeSettingsSummary>,
    services: Option<Arc<SurfaceServices>>,
    settings: Option<Settings>,
    error: Option<String>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchField {
    Query,
    Decks,
    Tags,
    Limit,
    Semantic,
    Fts,
    Verbose,
    Run,
    Results,
}

impl SearchField {
    const ORDER: [SearchField; 9] = [
        SearchField::Query,
        SearchField::Decks,
        SearchField::Tags,
        SearchField::Limit,
        SearchField::Semantic,
        SearchField::Fts,
        SearchField::Verbose,
        SearchField::Run,
        SearchField::Results,
    ];

    fn next(self) -> Self {
        next_in(Self::ORDER, self)
    }

    fn previous(self) -> Self {
        previous_in(Self::ORDER, self)
    }
}

#[derive(Debug, Clone)]
struct SearchState {
    query: String,
    deck_names: String,
    tags: String,
    limit: String,
    semantic_only: bool,
    fts_only: bool,
    verbose: bool,
    selected_field: SearchField,
    editing: bool,
    result: Option<HybridSearchResult>,
    error: Option<String>,
    selected_result: usize,
}

impl Default for SearchState {
    fn default() -> Self {
        Self {
            query: String::new(),
            deck_names: String::new(),
            tags: String::new(),
            limit: "10".to_string(),
            semantic_only: false,
            fts_only: false,
            verbose: false,
            selected_field: SearchField::Query,
            editing: false,
            result: None,
            error: None,
            selected_result: 0,
        }
    }
}

impl SearchState {
    fn current_text_field_mut(&mut self) -> Option<&mut String> {
        match self.selected_field {
            SearchField::Query => Some(&mut self.query),
            SearchField::Decks => Some(&mut self.deck_names),
            SearchField::Tags => Some(&mut self.tags),
            SearchField::Limit => Some(&mut self.limit),
            _ => None,
        }
    }

    fn next_field(&mut self) {
        self.selected_field = self.selected_field.next();
    }

    fn previous_field(&mut self) {
        self.selected_field = self.selected_field.previous();
    }

    fn request(&self) -> SearchRequest {
        SearchRequest {
            query: self.query.clone(),
            deck_names: split_csv(&self.deck_names),
            tags: split_csv(&self.tags),
            limit: self.limit.parse::<usize>().unwrap_or(10).max(1),
            semantic_only: self.semantic_only,
            fts_only: self.fts_only,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TopicsTab {
    Tree,
    Coverage,
    Gaps,
    WeakNotes,
    Duplicates,
}

impl TopicsTab {
    const ALL: [TopicsTab; 5] = [
        TopicsTab::Tree,
        TopicsTab::Coverage,
        TopicsTab::Gaps,
        TopicsTab::WeakNotes,
        TopicsTab::Duplicates,
    ];

    fn title(&self) -> &'static str {
        match self {
            Self::Tree => "Tree",
            Self::Coverage => "Coverage",
            Self::Gaps => "Gaps",
            Self::WeakNotes => "Weak Notes",
            Self::Duplicates => "Duplicates",
        }
    }

    fn next(self) -> Self {
        next_in(Self::ALL, self)
    }

    fn previous(self) -> Self {
        previous_in(Self::ALL, self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TopicsField {
    Tab,
    InputA,
    InputB,
    Toggle,
    InputC,
    InputD,
    Run,
}

impl TopicsField {
    const ORDER: [TopicsField; 7] = [
        TopicsField::Tab,
        TopicsField::InputA,
        TopicsField::InputB,
        TopicsField::Toggle,
        TopicsField::InputC,
        TopicsField::InputD,
        TopicsField::Run,
    ];

    fn next(self) -> Self {
        next_in(Self::ORDER, self)
    }

    fn previous(self) -> Self {
        previous_in(Self::ORDER, self)
    }
}

#[derive(Debug, Clone)]
struct TopicsState {
    tab: TopicsTab,
    selected_field: TopicsField,
    editing: bool,
    tree_root: String,
    coverage_topic: String,
    coverage_include_subtree: bool,
    gaps_topic: String,
    gaps_min_coverage: String,
    weak_topic: String,
    weak_limit: String,
    duplicates_threshold: String,
    duplicates_max: String,
    duplicates_decks: String,
    duplicates_tags: String,
    tree_result: Option<Vec<serde_json::Value>>,
    coverage_result: Option<TopicCoverage>,
    gaps_result: Option<Vec<TopicGap>>,
    weak_notes_result: Option<Vec<WeakNote>>,
    duplicates_result: Option<(Vec<DuplicateCluster>, DuplicateStats)>,
    error: Option<String>,
}

impl Default for TopicsState {
    fn default() -> Self {
        Self {
            tab: TopicsTab::Tree,
            selected_field: TopicsField::Tab,
            editing: false,
            tree_root: String::new(),
            coverage_topic: "rust".to_string(),
            coverage_include_subtree: true,
            gaps_topic: "rust".to_string(),
            gaps_min_coverage: "1".to_string(),
            weak_topic: "rust".to_string(),
            weak_limit: "20".to_string(),
            duplicates_threshold: "0.92".to_string(),
            duplicates_max: "50".to_string(),
            duplicates_decks: String::new(),
            duplicates_tags: String::new(),
            tree_result: None,
            coverage_result: None,
            gaps_result: None,
            weak_notes_result: None,
            duplicates_result: None,
            error: None,
        }
    }
}

impl TopicsState {
    fn next_field(&mut self) {
        self.selected_field = self.selected_field.next();
    }

    fn previous_field(&mut self) {
        self.selected_field = self.selected_field.previous();
    }

    fn current_text_field_mut(&mut self) -> Option<&mut String> {
        match (self.tab, self.selected_field) {
            (TopicsTab::Tree, TopicsField::InputA) => Some(&mut self.tree_root),
            (TopicsTab::Coverage, TopicsField::InputA) => Some(&mut self.coverage_topic),
            (TopicsTab::Gaps, TopicsField::InputA) => Some(&mut self.gaps_topic),
            (TopicsTab::Gaps, TopicsField::InputB) => Some(&mut self.gaps_min_coverage),
            (TopicsTab::WeakNotes, TopicsField::InputA) => Some(&mut self.weak_topic),
            (TopicsTab::WeakNotes, TopicsField::InputB) => Some(&mut self.weak_limit),
            (TopicsTab::Duplicates, TopicsField::InputA) => Some(&mut self.duplicates_threshold),
            (TopicsTab::Duplicates, TopicsField::InputB) => Some(&mut self.duplicates_max),
            (TopicsTab::Duplicates, TopicsField::InputC) => Some(&mut self.duplicates_decks),
            (TopicsTab::Duplicates, TopicsField::InputD) => Some(&mut self.duplicates_tags),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkflowTab {
    Sync,
    Index,
    Generate,
    Validate,
    Obsidian,
    TagAudit,
}

impl WorkflowTab {
    const ALL: [WorkflowTab; 6] = [
        WorkflowTab::Sync,
        WorkflowTab::Index,
        WorkflowTab::Generate,
        WorkflowTab::Validate,
        WorkflowTab::Obsidian,
        WorkflowTab::TagAudit,
    ];

    fn title(&self) -> &'static str {
        match self {
            Self::Sync => "Sync",
            Self::Index => "Index",
            Self::Generate => "Generate",
            Self::Validate => "Validate",
            Self::Obsidian => "Obsidian",
            Self::TagAudit => "Tag Audit",
        }
    }

    fn next(self) -> Self {
        next_in(Self::ALL, self)
    }

    fn previous(self) -> Self {
        previous_in(Self::ALL, self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkflowField {
    Tab,
    InputA,
    InputB,
    ToggleA,
    ToggleB,
    ToggleC,
    InputC,
    Run,
}

impl WorkflowField {
    const ORDER: [WorkflowField; 8] = [
        WorkflowField::Tab,
        WorkflowField::InputA,
        WorkflowField::InputB,
        WorkflowField::ToggleA,
        WorkflowField::ToggleB,
        WorkflowField::ToggleC,
        WorkflowField::InputC,
        WorkflowField::Run,
    ];

    fn next(self) -> Self {
        next_in(Self::ORDER, self)
    }

    fn previous(self) -> Self {
        previous_in(Self::ORDER, self)
    }
}

#[derive(Debug, Clone)]
struct WorkflowsState {
    tab: WorkflowTab,
    selected_field: WorkflowField,
    editing: bool,
    sync_source: String,
    sync_run_migrations: bool,
    sync_run_index: bool,
    sync_force_reindex: bool,
    index_force_reindex: bool,
    generate_file: String,
    validate_file: String,
    validate_quality: bool,
    obsidian_vault: String,
    obsidian_source_dirs: String,
    obsidian_dry_run: bool,
    tag_audit_file: String,
    tag_audit_fix: bool,
    sync_result: Option<SyncExecutionSummary>,
    index_result: Option<IndexExecutionSummary>,
    generate_result: Option<GeneratePreview>,
    validate_result: Option<ValidationSummary>,
    obsidian_result: Option<ObsidianScanPreview>,
    tag_audit_result: Option<TagAuditSummary>,
    error: Option<String>,
}

impl Default for WorkflowsState {
    fn default() -> Self {
        Self {
            tab: WorkflowTab::Sync,
            selected_field: WorkflowField::Tab,
            editing: false,
            sync_source: String::new(),
            sync_run_migrations: true,
            sync_run_index: true,
            sync_force_reindex: false,
            index_force_reindex: false,
            generate_file: String::new(),
            validate_file: String::new(),
            validate_quality: false,
            obsidian_vault: String::new(),
            obsidian_source_dirs: String::new(),
            obsidian_dry_run: true,
            tag_audit_file: String::new(),
            tag_audit_fix: false,
            sync_result: None,
            index_result: None,
            generate_result: None,
            validate_result: None,
            obsidian_result: None,
            tag_audit_result: None,
            error: None,
        }
    }
}

impl WorkflowsState {
    fn next_field(&mut self) {
        self.selected_field = self.selected_field.next();
    }

    fn previous_field(&mut self) {
        self.selected_field = self.selected_field.previous();
    }

    fn current_text_field_mut(&mut self) -> Option<&mut String> {
        match (self.tab, self.selected_field) {
            (WorkflowTab::Sync, WorkflowField::InputA) => Some(&mut self.sync_source),
            (WorkflowTab::Generate, WorkflowField::InputA) => Some(&mut self.generate_file),
            (WorkflowTab::Validate, WorkflowField::InputA) => Some(&mut self.validate_file),
            (WorkflowTab::Obsidian, WorkflowField::InputA) => Some(&mut self.obsidian_vault),
            (WorkflowTab::Obsidian, WorkflowField::InputB) => Some(&mut self.obsidian_source_dirs),
            (WorkflowTab::TagAudit, WorkflowField::InputA) => Some(&mut self.tag_audit_file),
            _ => None,
        }
    }
}

#[derive(Clone)]
struct AppState {
    screen: Screen,
    navigation_index: usize,
    focus_region: FocusRegion,
    modal: Option<ModalState>,
    runtime: RuntimeState,
    activity: VecDeque<String>,
    busy_label: Option<String>,
    progress: Option<SurfaceProgressEvent>,
    last_result_summary: Option<String>,
    search: SearchState,
    topics: TopicsState,
    workflows: WorkflowsState,
    should_quit: bool,
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
    fn push_activity(&mut self, message: String) {
        self.activity.push_front(message);
        while self.activity.len() > 8 {
            let _ = self.activity.pop_back();
        }
    }

    fn current_screen(&self) -> Screen {
        Screen::ALL[self.navigation_index]
    }

    fn set_screen(&mut self, screen: Screen) {
        self.screen = screen;
        self.navigation_index = Screen::ALL
            .iter()
            .position(|candidate| *candidate == screen)
            .unwrap_or(0);
    }

    fn runtime_handles(&self) -> Option<RuntimeHandles> {
        self.runtime.services.as_deref().map(RuntimeHandles::from)
    }

    fn apply_event(&mut self, event: AppEvent) {
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
                        self.last_result_summary = Some(format!(
                            "sync upserted {} notes",
                            result.sync.notes_upserted
                        ));
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

enum TaskResult {
    Bootstrap(RuntimeBootstrap),
    Search(HybridSearchResult),
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

enum AppEvent {
    TaskFinished {
        label: String,
        result: Box<Result<TaskResult, String>>,
    },
    Progress(SurfaceProgressEvent),
    Quit,
}

fn handle_key_event(app: &mut AppState, key: KeyEvent, tx: &mpsc::UnboundedSender<AppEvent>) {
    if app.modal.is_some() {
        match key.code {
            KeyCode::Esc | KeyCode::Char('?') | KeyCode::Enter => app.modal = None,
            KeyCode::Char('q') => app.should_quit = true,
            _ => {}
        }
        return;
    }

    if matches!(key.code, KeyCode::Char('q')) && key.modifiers.is_empty() {
        app.should_quit = true;
        return;
    }

    if matches!(key.code, KeyCode::Char('?')) {
        app.modal = Some(ModalState::Help);
        return;
    }

    if matches!(key.code, KeyCode::Char('/')) {
        app.set_screen(Screen::Search);
        app.focus_region = FocusRegion::Content;
        app.search.selected_field = SearchField::Query;
        app.search.editing = true;
        return;
    }

    if matches!(key.code, KeyCode::Esc) {
        match app.focus_region {
            FocusRegion::Content => {
                match app.screen {
                    Screen::Search => app.search.editing = false,
                    Screen::Topics => app.topics.editing = false,
                    Screen::Workflows => app.workflows.editing = false,
                    Screen::Home => {}
                }
                app.focus_region = FocusRegion::Navigation;
            }
            FocusRegion::Navigation => app.set_screen(Screen::Home),
        }
        return;
    }

    if key.code == KeyCode::Tab {
        match app.focus_region {
            FocusRegion::Navigation => app.focus_region = FocusRegion::Content,
            FocusRegion::Content => match app.screen {
                Screen::Search => app.search.next_field(),
                Screen::Topics => app.topics.next_field(),
                Screen::Workflows => app.workflows.next_field(),
                Screen::Home => {}
            },
        }
        return;
    }

    if key.code == KeyCode::BackTab {
        match app.focus_region {
            FocusRegion::Content => app.focus_region = FocusRegion::Navigation,
            FocusRegion::Navigation => {}
        }
        return;
    }

    match app.focus_region {
        FocusRegion::Navigation => handle_navigation_input(app, key),
        FocusRegion::Content => handle_content_input(app, key, tx),
    }
}

fn handle_navigation_input(app: &mut AppState, key: KeyEvent) {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => {
            app.navigation_index = app.navigation_index.saturating_sub(1);
            app.set_screen(Screen::ALL[app.navigation_index]);
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.navigation_index = (app.navigation_index + 1).min(Screen::ALL.len() - 1);
            app.set_screen(Screen::ALL[app.navigation_index]);
        }
        KeyCode::Enter => {
            app.focus_region = FocusRegion::Content;
        }
        KeyCode::Char('r') if matches!(app.runtime.status, RuntimeStatus::Error) => {
            app.runtime.status = RuntimeStatus::Bootstrapping;
            app.runtime.error = None;
        }
        _ => {}
    }
}

fn handle_content_input(app: &mut AppState, key: KeyEvent, tx: &mpsc::UnboundedSender<AppEvent>) {
    match app.screen {
        Screen::Home => handle_home_input(app, key, tx),
        Screen::Search => handle_search_input(app, key, tx),
        Screen::Topics => handle_topics_input(app, key, tx),
        Screen::Workflows => handle_workflows_input(app, key, tx),
    }
}

fn handle_home_input(app: &mut AppState, key: KeyEvent, tx: &mpsc::UnboundedSender<AppEvent>) {
    match key.code {
        KeyCode::Char('r') => {
            app.runtime.status = RuntimeStatus::Bootstrapping;
            app.runtime.error = None;
            app.busy_label = Some("runtime bootstrap".to_string());
            app.push_activity("retrying runtime bootstrap".to_string());
            start_bootstrap(tx.clone(), true);
        }
        KeyCode::Char('s') => app.set_screen(Screen::Search),
        KeyCode::Char('t') => app.set_screen(Screen::Topics),
        KeyCode::Char('w') => app.set_screen(Screen::Workflows),
        _ => {}
    }
}

fn handle_search_input(app: &mut AppState, key: KeyEvent, tx: &mpsc::UnboundedSender<AppEvent>) {
    if app.search.editing {
        if handle_text_edit(app.search.current_text_field_mut(), key) {
            return;
        }
        app.search.editing = false;
        if matches!(key.code, KeyCode::Enter | KeyCode::Esc) {
            return;
        }
    }

    match key.code {
        KeyCode::Up | KeyCode::Char('k') => app.search.previous_field(),
        KeyCode::Down | KeyCode::Char('j') => app.search.next_field(),
        KeyCode::Left | KeyCode::Char('h') => {
            app.search.selected_result = app.search.selected_result.saturating_sub(1);
        }
        KeyCode::Right | KeyCode::Char('l') => {
            if let Some(result) = &app.search.result {
                if !result.results.is_empty() {
                    app.search.selected_result =
                        (app.search.selected_result + 1).min(result.results.len() - 1);
                }
            }
        }
        KeyCode::Enter => match app.search.selected_field {
            SearchField::Query | SearchField::Decks | SearchField::Tags | SearchField::Limit => {
                app.search.editing = true;
            }
            SearchField::Semantic => app.search.semantic_only = !app.search.semantic_only,
            SearchField::Fts => app.search.fts_only = !app.search.fts_only,
            SearchField::Verbose => app.search.verbose = !app.search.verbose,
            SearchField::Run => {
                if let Some(handles) = app.runtime_handles() {
                    let request = app.search.request();
                    app.busy_label = Some("search".to_string());
                    spawn_task(tx.clone(), "search", move || async move {
                        usecases::search(handles, request)
                            .await
                            .map(TaskResult::Search)
                    });
                }
            }
            SearchField::Results => {}
        },
        KeyCode::Char(' ') => match app.search.selected_field {
            SearchField::Semantic => app.search.semantic_only = !app.search.semantic_only,
            SearchField::Fts => app.search.fts_only = !app.search.fts_only,
            SearchField::Verbose => app.search.verbose = !app.search.verbose,
            _ => {}
        },
        _ => {}
    }
}

fn handle_topics_input(app: &mut AppState, key: KeyEvent, tx: &mpsc::UnboundedSender<AppEvent>) {
    if app.topics.editing {
        if handle_text_edit(app.topics.current_text_field_mut(), key) {
            return;
        }
        app.topics.editing = false;
        if matches!(key.code, KeyCode::Enter | KeyCode::Esc) {
            return;
        }
    }

    match key.code {
        KeyCode::Up | KeyCode::Char('k') => app.topics.previous_field(),
        KeyCode::Down | KeyCode::Char('j') => app.topics.next_field(),
        KeyCode::Left | KeyCode::Char('h') if app.topics.selected_field == TopicsField::Tab => {
            app.topics.tab = app.topics.tab.previous();
        }
        KeyCode::Right | KeyCode::Char('l') if app.topics.selected_field == TopicsField::Tab => {
            app.topics.tab = app.topics.tab.next();
        }
        KeyCode::Enter => match app.topics.selected_field {
            TopicsField::Tab => app.topics.tab = app.topics.tab.next(),
            TopicsField::InputA
            | TopicsField::InputB
            | TopicsField::InputC
            | TopicsField::InputD => {
                app.topics.editing = app.topics.current_text_field_mut().is_some();
            }
            TopicsField::Toggle if app.topics.tab == TopicsTab::Coverage => {
                app.topics.coverage_include_subtree = !app.topics.coverage_include_subtree;
            }
            TopicsField::Run => run_topics_task(app, tx),
            TopicsField::Toggle => {}
        },
        KeyCode::Char(' ') if app.topics.selected_field == TopicsField::Toggle => {
            if app.topics.tab == TopicsTab::Coverage {
                app.topics.coverage_include_subtree = !app.topics.coverage_include_subtree;
            }
        }
        _ => {}
    }
}

fn handle_workflows_input(app: &mut AppState, key: KeyEvent, tx: &mpsc::UnboundedSender<AppEvent>) {
    if app.workflows.editing {
        if handle_text_edit(app.workflows.current_text_field_mut(), key) {
            return;
        }
        app.workflows.editing = false;
        if matches!(key.code, KeyCode::Enter | KeyCode::Esc) {
            return;
        }
    }

    match key.code {
        KeyCode::Up | KeyCode::Char('k') => app.workflows.previous_field(),
        KeyCode::Down | KeyCode::Char('j') => app.workflows.next_field(),
        KeyCode::Left | KeyCode::Char('h')
            if app.workflows.selected_field == WorkflowField::Tab =>
        {
            app.workflows.tab = app.workflows.tab.previous();
        }
        KeyCode::Right | KeyCode::Char('l')
            if app.workflows.selected_field == WorkflowField::Tab =>
        {
            app.workflows.tab = app.workflows.tab.next();
        }
        KeyCode::Enter => match app.workflows.selected_field {
            WorkflowField::Tab => app.workflows.tab = app.workflows.tab.next(),
            WorkflowField::InputA | WorkflowField::InputB | WorkflowField::InputC => {
                app.workflows.editing = app.workflows.current_text_field_mut().is_some();
            }
            WorkflowField::ToggleA | WorkflowField::ToggleB | WorkflowField::ToggleC => {
                toggle_workflow_checkbox(&mut app.workflows);
            }
            WorkflowField::Run => run_workflow_task(app, tx),
        },
        KeyCode::Char(' ') => toggle_workflow_checkbox(&mut app.workflows),
        _ => {}
    }
}

fn handle_text_edit(target: Option<&mut String>, key: KeyEvent) -> bool {
    let Some(target) = target else {
        return false;
    };

    match key.code {
        KeyCode::Char(ch) if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT => {
            target.push(ch);
            true
        }
        KeyCode::Backspace => {
            let _ = target.pop();
            true
        }
        KeyCode::Enter | KeyCode::Esc => false,
        _ => true,
    }
}

fn toggle_workflow_checkbox(workflows: &mut WorkflowsState) {
    match (workflows.tab, workflows.selected_field) {
        (WorkflowTab::Sync, WorkflowField::ToggleA) => {
            workflows.sync_run_migrations = !workflows.sync_run_migrations;
        }
        (WorkflowTab::Sync, WorkflowField::ToggleB) => {
            workflows.sync_run_index = !workflows.sync_run_index;
        }
        (WorkflowTab::Sync, WorkflowField::ToggleC) => {
            workflows.sync_force_reindex = !workflows.sync_force_reindex;
        }
        (WorkflowTab::Index, WorkflowField::ToggleA) => {
            workflows.index_force_reindex = !workflows.index_force_reindex;
        }
        (WorkflowTab::Validate, WorkflowField::ToggleA) => {
            workflows.validate_quality = !workflows.validate_quality;
        }
        (WorkflowTab::Obsidian, WorkflowField::ToggleA) => {
            workflows.obsidian_dry_run = !workflows.obsidian_dry_run;
        }
        (WorkflowTab::TagAudit, WorkflowField::ToggleA) => {
            workflows.tag_audit_fix = !workflows.tag_audit_fix;
        }
        _ => {}
    }
}

fn run_topics_task(app: &mut AppState, tx: &mpsc::UnboundedSender<AppEvent>) {
    let Some(handles) = app.runtime_handles() else {
        return;
    };

    match app.topics.tab {
        TopicsTab::Tree => {
            let request = TopicsTreeRequest {
                root_path: (!app.topics.tree_root.trim().is_empty())
                    .then(|| app.topics.tree_root.trim().to_string()),
            };
            app.busy_label = Some("topics tree".to_string());
            spawn_task(tx.clone(), "topics tree", move || async move {
                usecases::topics_tree(handles, request)
                    .await
                    .map(TaskResult::TopicsTree)
            });
        }
        TopicsTab::Coverage => {
            let request = CoverageRequest {
                topic: app.topics.coverage_topic.clone(),
                include_subtree: app.topics.coverage_include_subtree,
            };
            app.busy_label = Some("coverage".to_string());
            spawn_task(tx.clone(), "coverage", move || async move {
                usecases::coverage(handles, request)
                    .await
                    .map(TaskResult::Coverage)
            });
        }
        TopicsTab::Gaps => {
            let request = GapsRequest {
                topic: app.topics.gaps_topic.clone(),
                min_coverage: app.topics.gaps_min_coverage.parse::<i64>().unwrap_or(1),
            };
            app.busy_label = Some("gaps".to_string());
            spawn_task(tx.clone(), "gaps", move || async move {
                usecases::gaps(handles, request).await.map(TaskResult::Gaps)
            });
        }
        TopicsTab::WeakNotes => {
            let request = WeakNotesRequest {
                topic: app.topics.weak_topic.clone(),
                limit: app.topics.weak_limit.parse::<i64>().unwrap_or(20),
            };
            app.busy_label = Some("weak notes".to_string());
            spawn_task(tx.clone(), "weak notes", move || async move {
                usecases::weak_notes(handles, request)
                    .await
                    .map(TaskResult::WeakNotes)
            });
        }
        TopicsTab::Duplicates => {
            let request = DuplicatesRequest {
                threshold: app
                    .topics
                    .duplicates_threshold
                    .parse::<f64>()
                    .unwrap_or(0.92),
                max: app.topics.duplicates_max.parse::<usize>().unwrap_or(50),
                deck_names: split_csv(&app.topics.duplicates_decks),
                tags: split_csv(&app.topics.duplicates_tags),
            };
            app.busy_label = Some("duplicates".to_string());
            spawn_task(tx.clone(), "duplicates", move || async move {
                usecases::duplicates(handles, request)
                    .await
                    .map(|(clusters, stats)| TaskResult::Duplicates(clusters, stats))
            });
        }
    }
}

fn run_workflow_task(app: &mut AppState, tx: &mpsc::UnboundedSender<AppEvent>) {
    let Some(handles) = app.runtime_handles() else {
        return;
    };

    match app.workflows.tab {
        WorkflowTab::Sync => {
            let request = SyncRequest {
                source: app.workflows.sync_source.clone().into(),
                run_migrations: app.workflows.sync_run_migrations,
                run_index: app.workflows.sync_run_index,
                force_reindex: app.workflows.sync_force_reindex,
            };
            let progress = progress_sink(tx.clone());
            app.busy_label = Some("sync".to_string());
            spawn_task(tx.clone(), "sync", move || async move {
                usecases::sync(handles, request, Some(progress))
                    .await
                    .map(TaskResult::Sync)
            });
        }
        WorkflowTab::Index => {
            let request = IndexRequest {
                force_reindex: app.workflows.index_force_reindex,
            };
            let progress = progress_sink(tx.clone());
            app.busy_label = Some("index".to_string());
            spawn_task(tx.clone(), "index", move || async move {
                usecases::index(handles, request, Some(progress))
                    .await
                    .map(TaskResult::Index)
            });
        }
        WorkflowTab::Generate => {
            let request = GenerateRequest {
                file: app.workflows.generate_file.clone().into(),
            };
            let service = Arc::clone(&handles.generate_preview);
            app.busy_label = Some("generate preview".to_string());
            spawn_task(tx.clone(), "generate preview", move || async move {
                usecases::generate_preview(service.as_ref(), &request).map(TaskResult::Generate)
            });
        }
        WorkflowTab::Validate => {
            let request = ValidateRequest {
                file: app.workflows.validate_file.clone().into(),
                include_quality: app.workflows.validate_quality,
            };
            let service = Arc::clone(&handles.validation);
            app.busy_label = Some("validate".to_string());
            spawn_task(tx.clone(), "validate", move || async move {
                usecases::validate(service.as_ref(), &request).map(TaskResult::Validate)
            });
        }
        WorkflowTab::Obsidian => {
            let request = ObsidianScanRequest {
                vault: app.workflows.obsidian_vault.clone().into(),
                source_dirs: split_csv(&app.workflows.obsidian_source_dirs),
                dry_run: app.workflows.obsidian_dry_run,
            };
            let progress = progress_sink(tx.clone());
            let service = Arc::clone(&handles.obsidian_scan);
            app.busy_label = Some("obsidian scan".to_string());
            spawn_task(tx.clone(), "obsidian scan", move || async move {
                usecases::obsidian_scan(service.as_ref(), &request, Some(progress))
                    .map(TaskResult::Obsidian)
            });
        }
        WorkflowTab::TagAudit => {
            let request = TagAuditRequest {
                file: app.workflows.tag_audit_file.clone().into(),
                apply_fixes: app.workflows.tag_audit_fix,
            };
            let service = Arc::clone(&handles.tag_audit);
            app.busy_label = Some("tag audit".to_string());
            spawn_task(tx.clone(), "tag audit", move || async move {
                usecases::tag_audit(service.as_ref(), &request).map(TaskResult::TagAudit)
            });
        }
    }
}

fn progress_sink(tx: mpsc::UnboundedSender<AppEvent>) -> SurfaceProgressSink {
    Arc::new(move |progress| {
        let _ = tx.send(AppEvent::Progress(progress));
    })
}

fn spawn_task<F, Fut>(tx: mpsc::UnboundedSender<AppEvent>, label: &'static str, task: F)
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = anyhow::Result<TaskResult>> + 'static,
{
    std::thread::spawn(move || {
        let result = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime.block_on(task()).map_err(|error| error.to_string()),
            Err(error) => Err(error.to_string()),
        };
        let _ = tx.send(AppEvent::TaskFinished {
            label: label.to_string(),
            result: Box::new(result),
        });
    });
}

fn render(frame: &mut Frame<'_>, app: &AppState) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(6)])
        .split(frame.area());
    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(20), Constraint::Min(20)])
        .split(layout[0]);

    render_navigation(frame, body[0], app);
    match app.screen {
        Screen::Home => render_home(frame, body[1], app),
        Screen::Search => render_search(frame, body[1], app),
        Screen::Topics => render_topics(frame, body[1], app),
        Screen::Workflows => render_workflows(frame, body[1], app),
    }
    render_status(frame, layout[1], app);

    if app.modal.is_some() {
        render_help_modal(frame);
    }
}

fn render_navigation(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let items: Vec<ListItem<'_>> = Screen::ALL
        .iter()
        .enumerate()
        .map(|(index, screen)| {
            let selected = app.navigation_index == index;
            let focused = app.focus_region == FocusRegion::Navigation && selected;
            let style = if focused {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else if selected {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(vec![Span::styled(screen.title(), style)]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .title("Screens")
            .borders(Borders::ALL)
            .border_style(region_border(app.focus_region == FocusRegion::Navigation)),
    );
    frame.render_widget(list, area);
}

fn render_home(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Min(6),
        ])
        .split(area);

    let status_lines = match app.runtime.status {
        RuntimeStatus::Bootstrapping => vec![
            Line::from("Runtime status: bootstrapping"),
            Line::from("Press `?` for help, `q` to quit."),
        ],
        RuntimeStatus::Ready => {
            let summary = app.runtime.summary.as_ref();
            vec![
                Line::from("Runtime status: ready"),
                Line::from(format!(
                    "Postgres: {}",
                    summary
                        .map(|item| item.postgres_url.as_str())
                        .unwrap_or("(unknown)")
                )),
                Line::from(format!(
                    "Qdrant: {}",
                    summary
                        .map(|item| item.qdrant_url.as_str())
                        .unwrap_or("(unknown)")
                )),
                Line::from(format!(
                    "Redis: {}",
                    summary
                        .map(|item| item.redis_url.as_str())
                        .unwrap_or("(unknown)")
                )),
                Line::from(format!(
                    "Embedding: {} / {}",
                    summary
                        .map(|item| item.embedding_provider.as_str())
                        .unwrap_or("(unknown)"),
                    summary
                        .map(|item| item.embedding_model.as_str())
                        .unwrap_or("(unknown)")
                )),
                Line::from(format!(
                    "Rerank enabled: {}",
                    summary.map(|item| item.rerank_enabled).unwrap_or(false)
                )),
            ]
        }
        RuntimeStatus::Error => vec![
            Line::from("Runtime status: failed"),
            Line::from(
                app.runtime
                    .error
                    .clone()
                    .unwrap_or_else(|| "unknown bootstrap error".to_string()),
            ),
            Line::from("Press `r` to retry bootstrap."),
        ],
    };

    let quick_actions = vec![
        Line::from("Quick actions"),
        Line::from("`s` open Search"),
        Line::from("`t` open Topics"),
        Line::from("`w` open Workflows"),
        Line::from("`r` retry bootstrap"),
    ];

    let last_result_lines = vec![
        Line::from(format!(
            "Last result: {}",
            app.last_result_summary
                .as_deref()
                .unwrap_or("no completed actions yet")
        )),
        Line::from(format!(
            "Busy: {}",
            app.busy_label.as_deref().unwrap_or("idle")
        )),
        Line::from(format!(
            "Progress: {}",
            app.progress
                .as_ref()
                .map(|progress| progress.message.as_str())
                .unwrap_or("n/a")
        )),
    ];

    frame.render_widget(
        Paragraph::new(Text::from(status_lines))
            .block(Block::default().title("Bootstrap").borders(Borders::ALL))
            .wrap(Wrap { trim: true }),
        chunks[0],
    );
    frame.render_widget(
        Paragraph::new(Text::from(quick_actions))
            .block(
                Block::default()
                    .title("Quick Actions")
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: true }),
        chunks[1],
    );
    frame.render_widget(
        Paragraph::new(Text::from(last_result_lines))
            .block(Block::default().title("Session").borders(Borders::ALL))
            .wrap(Wrap { trim: true }),
        chunks[2],
    );
}

fn render_search(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(10)])
        .split(area);
    let form = search_form_lines(&app.search, app.focus_region == FocusRegion::Content);
    frame.render_widget(
        Paragraph::new(Text::from(form))
            .block(Block::default().title("Search Form").borders(Borders::ALL))
            .wrap(Wrap { trim: true }),
        sections[0],
    );

    let detail = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(36), Constraint::Min(20)])
        .split(sections[1]);

    let results: Vec<ListItem<'_>> = app
        .search
        .result
        .as_ref()
        .map(|result| {
            result
                .results
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    let style = if index == app.search.selected_result {
                        Style::default().fg(Color::Yellow)
                    } else {
                        Style::default()
                    };
                    ListItem::new(Line::from(vec![Span::styled(
                        format!("{}. note={} {:.3}", index + 1, item.note_id, item.rrf_score),
                        style,
                    )]))
                })
                .collect()
        })
        .unwrap_or_else(|| vec![ListItem::new("No search results yet")]);

    frame.render_widget(
        List::new(results).block(Block::default().title("Results").borders(Borders::ALL)),
        detail[0],
    );
    frame.render_widget(
        Paragraph::new(Text::from(search_detail_lines(&app.search)))
            .block(Block::default().title("Detail").borders(Borders::ALL))
            .wrap(Wrap { trim: true }),
        detail[1],
    );
}

fn render_topics(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(10)])
        .split(area);
    let titles = TopicsTab::ALL
        .iter()
        .map(|tab| Line::from(tab.title()))
        .collect::<Vec<_>>();
    frame.render_widget(
        Tabs::new(titles)
            .select(
                TopicsTab::ALL
                    .iter()
                    .position(|tab| *tab == app.topics.tab)
                    .unwrap_or(0),
            )
            .block(Block::default().title("Topic Views").borders(Borders::ALL))
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        Rect {
            x: sections[0].x,
            y: sections[0].y,
            width: sections[0].width,
            height: 3,
        },
    );
    frame.render_widget(
        Paragraph::new(Text::from(topics_form_lines(&app.topics)))
            .block(
                Block::default()
                    .title("Topic Controls")
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: true }),
        Rect {
            x: sections[0].x,
            y: sections[0].y + 3,
            width: sections[0].width,
            height: sections[0].height.saturating_sub(3),
        },
    );
    frame.render_widget(
        Paragraph::new(Text::from(topics_result_lines(&app.topics)))
            .block(Block::default().title("Topic Result").borders(Borders::ALL))
            .wrap(Wrap { trim: true }),
        sections[1],
    );
}

fn render_workflows(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(11), Constraint::Min(10)])
        .split(area);
    let titles = WorkflowTab::ALL
        .iter()
        .map(|tab| Line::from(tab.title()))
        .collect::<Vec<_>>();
    frame.render_widget(
        Tabs::new(titles)
            .select(
                WorkflowTab::ALL
                    .iter()
                    .position(|tab| *tab == app.workflows.tab)
                    .unwrap_or(0),
            )
            .block(
                Block::default()
                    .title("Workflow Views")
                    .borders(Borders::ALL),
            )
            .highlight_style(
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        Rect {
            x: sections[0].x,
            y: sections[0].y,
            width: sections[0].width,
            height: 3,
        },
    );
    frame.render_widget(
        Paragraph::new(Text::from(workflow_form_lines(&app.workflows)))
            .block(
                Block::default()
                    .title("Workflow Controls")
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: true }),
        Rect {
            x: sections[0].x,
            y: sections[0].y + 3,
            width: sections[0].width,
            height: sections[0].height.saturating_sub(3),
        },
    );
    frame.render_widget(
        Paragraph::new(Text::from(workflow_result_lines(&app.workflows)))
            .block(
                Block::default()
                    .title("Workflow Result")
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: true }),
        sections[1],
    );
}

fn render_status(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(3)])
        .split(area);
    let status_line = Line::from(vec![
        Span::styled(
            format!("screen={} ", app.current_screen().title()),
            Style::default().fg(Color::Cyan),
        ),
        Span::raw(format!(
            "focus={} ",
            match app.focus_region {
                FocusRegion::Navigation => "nav",
                FocusRegion::Content => "content",
            }
        )),
        Span::raw(format!(
            "busy={} ",
            app.busy_label.as_deref().unwrap_or("idle")
        )),
        Span::raw(format!(
            "progress={} ",
            app.progress
                .as_ref()
                .map(|progress| progress.message.as_str())
                .unwrap_or("n/a")
        )),
    ]);
    frame.render_widget(
        Paragraph::new(Text::from(vec![status_line]))
            .block(Block::default().title("Status").borders(Borders::ALL)),
        chunks[0],
    );
    let activity: Vec<Line<'_>> = app
        .activity
        .iter()
        .take(4)
        .map(|entry| Line::from(entry.clone()))
        .collect();
    frame.render_widget(
        Paragraph::new(Text::from(activity))
            .block(Block::default().title("Activity").borders(Borders::ALL))
            .wrap(Wrap { trim: true }),
        chunks[1],
    );
}

fn render_help_modal(frame: &mut Frame<'_>) {
    let area = centered_rect(70, 50, frame.area());
    let content = Paragraph::new(Text::from(vec![
        Line::from("Keybindings"),
        Line::from("Tab: move focus within the current screen"),
        Line::from("Shift+Tab: move focus back to navigation"),
        Line::from("Up/Down or j/k: move selection"),
        Line::from("Left/Right or h/l: change tabs and result selection"),
        Line::from("Enter: edit, toggle, or run"),
        Line::from("/: jump to search query"),
        Line::from("Esc: back out"),
        Line::from("q: quit"),
    ]))
    .block(Block::default().title("Help").borders(Borders::ALL))
    .wrap(Wrap { trim: true });
    frame.render_widget(Clear, area);
    frame.render_widget(content, area);
}

fn search_form_lines(state: &SearchState, content_focused: bool) -> Vec<Line<'static>> {
    vec![
        field_line(
            state.selected_field == SearchField::Query && content_focused,
            format!(
                "query: {}",
                display_field_value(&state.query, state.editing)
            ),
        ),
        field_line(
            state.selected_field == SearchField::Decks && content_focused,
            format!("decks(csv): {}", empty_placeholder(&state.deck_names)),
        ),
        field_line(
            state.selected_field == SearchField::Tags && content_focused,
            format!("tags(csv): {}", empty_placeholder(&state.tags)),
        ),
        field_line(
            state.selected_field == SearchField::Limit && content_focused,
            format!("limit: {}", state.limit),
        ),
        field_line(
            state.selected_field == SearchField::Semantic && content_focused,
            format!("semantic only: {}", checkbox(state.semantic_only)),
        ),
        field_line(
            state.selected_field == SearchField::Fts && content_focused,
            format!("fts only: {}", checkbox(state.fts_only)),
        ),
        field_line(
            state.selected_field == SearchField::Verbose && content_focused,
            format!("verbose detail: {}", checkbox(state.verbose)),
        ),
        field_line(
            state.selected_field == SearchField::Run && content_focused,
            "[ run search ]".to_string(),
        ),
    ]
}

fn search_detail_lines(state: &SearchState) -> Vec<Line<'static>> {
    if let Some(error) = &state.error {
        return vec![Line::from(format!("error: {error}"))];
    }
    let Some(result) = &state.result else {
        return vec![Line::from("Run a search to inspect results.")];
    };
    if result.results.is_empty() {
        return vec![Line::from("No results.")];
    }
    let item = &result.results[state.selected_result.min(result.results.len() - 1)];
    let mut lines = vec![
        Line::from(format!("query: {}", result.query)),
        Line::from(format!("note_id: {}", item.note_id)),
        Line::from(format!("score: {:.4}", item.rrf_score)),
        Line::from(format!("sources: {}", item.sources().join(","))),
        Line::from(format!(
            "headline: {}",
            item.headline.as_deref().unwrap_or("(no headline)")
        )),
    ];
    if state.verbose {
        lines.push(Line::from(format!(
            "semantic={:?} fts={:?} rerank={:?}",
            item.semantic_score, item.fts_score, item.rerank_score
        )));
    }
    lines
}

fn topics_form_lines(state: &TopicsState) -> Vec<Line<'static>> {
    match state.tab {
        TopicsTab::Tree => vec![
            field_line(
                state.selected_field == TopicsField::InputA,
                format!("root path: {}", empty_placeholder(&state.tree_root)),
            ),
            field_line(
                state.selected_field == TopicsField::Run,
                "[ load tree ]".to_string(),
            ),
        ],
        TopicsTab::Coverage => vec![
            field_line(
                state.selected_field == TopicsField::InputA,
                format!("topic: {}", empty_placeholder(&state.coverage_topic)),
            ),
            field_line(
                state.selected_field == TopicsField::Toggle,
                format!(
                    "include subtree: {}",
                    checkbox(state.coverage_include_subtree)
                ),
            ),
            field_line(
                state.selected_field == TopicsField::Run,
                "[ run coverage ]".to_string(),
            ),
        ],
        TopicsTab::Gaps => vec![
            field_line(
                state.selected_field == TopicsField::InputA,
                format!("topic: {}", empty_placeholder(&state.gaps_topic)),
            ),
            field_line(
                state.selected_field == TopicsField::InputB,
                format!("min coverage: {}", state.gaps_min_coverage),
            ),
            field_line(
                state.selected_field == TopicsField::Run,
                "[ find gaps ]".to_string(),
            ),
        ],
        TopicsTab::WeakNotes => vec![
            field_line(
                state.selected_field == TopicsField::InputA,
                format!("topic: {}", empty_placeholder(&state.weak_topic)),
            ),
            field_line(
                state.selected_field == TopicsField::InputB,
                format!("limit: {}", state.weak_limit),
            ),
            field_line(
                state.selected_field == TopicsField::Run,
                "[ list weak notes ]".to_string(),
            ),
        ],
        TopicsTab::Duplicates => vec![
            field_line(
                state.selected_field == TopicsField::InputA,
                format!("threshold: {}", state.duplicates_threshold),
            ),
            field_line(
                state.selected_field == TopicsField::InputB,
                format!("max clusters: {}", state.duplicates_max),
            ),
            field_line(
                state.selected_field == TopicsField::InputC,
                format!(
                    "deck filters(csv): {}",
                    empty_placeholder(&state.duplicates_decks)
                ),
            ),
            field_line(
                state.selected_field == TopicsField::InputD,
                format!(
                    "tag filters(csv): {}",
                    empty_placeholder(&state.duplicates_tags)
                ),
            ),
            field_line(
                state.selected_field == TopicsField::Run,
                "[ find duplicates ]".to_string(),
            ),
        ],
    }
}

fn topics_result_lines(state: &TopicsState) -> Vec<Line<'static>> {
    if let Some(error) = &state.error {
        return vec![Line::from(format!("error: {error}"))];
    }
    match state.tab {
        TopicsTab::Tree => state
            .tree_result
            .as_ref()
            .map(|result| {
                let pretty =
                    serde_json::to_string_pretty(result).unwrap_or_else(|_| "[]".to_string());
                pretty
                    .lines()
                    .take(24)
                    .map(|line| Line::from(line.to_string()))
                    .collect()
            })
            .unwrap_or_else(|| vec![Line::from("Run tree view to inspect taxonomy.")]),
        TopicsTab::Coverage => state
            .coverage_result
            .as_ref()
            .map(|result| {
                vec![
                    Line::from(format!("topic: {} ({})", result.path, result.label)),
                    Line::from(format!("note_count: {}", result.note_count)),
                    Line::from(format!("subtree_count: {}", result.subtree_count)),
                    Line::from(format!(
                        "covered_children: {}/{}",
                        result.covered_children, result.child_count
                    )),
                    Line::from(format!("mature_count: {}", result.mature_count)),
                    Line::from(format!("avg_confidence: {:.3}", result.avg_confidence)),
                    Line::from(format!("weak_notes: {}", result.weak_notes)),
                ]
            })
            .unwrap_or_else(|| vec![Line::from("Run coverage to inspect a topic.")]),
        TopicsTab::Gaps => state
            .gaps_result
            .as_ref()
            .map(|result| {
                if result.is_empty() {
                    return vec![Line::from("No gaps found.")];
                }
                result
                    .iter()
                    .take(24)
                    .map(|gap| {
                        Line::from(format!(
                            "{} [{}] notes={} threshold={}",
                            gap.path,
                            serde_json::to_string(&gap.gap_type)
                                .unwrap_or_else(|_| "\"unknown\"".to_string()),
                            gap.note_count,
                            gap.threshold
                        ))
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![Line::from("Run gap analysis to inspect missing coverage.")]),
        TopicsTab::WeakNotes => state
            .weak_notes_result
            .as_ref()
            .map(|result| {
                if result.is_empty() {
                    return vec![Line::from("No weak notes found.")];
                }
                result
                    .iter()
                    .take(20)
                    .flat_map(|note| {
                        [
                            Line::from(format!(
                                "note={} confidence={:.3} lapses={} fail_rate={:?}",
                                note.note_id, note.confidence, note.lapses, note.fail_rate
                            )),
                            Line::from(note.normalized_text.clone()),
                        ]
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![Line::from("Run weak note analysis to inspect signals.")]),
        TopicsTab::Duplicates => state
            .duplicates_result
            .as_ref()
            .map(|(clusters, stats)| {
                let mut lines = vec![
                    Line::from(format!("clusters: {}", stats.clusters_found)),
                    Line::from(format!("notes_scanned: {}", stats.notes_scanned)),
                    Line::from(format!("total_duplicates: {}", stats.total_duplicates)),
                ];
                lines.extend(clusters.iter().take(20).map(|cluster| {
                    Line::from(format!(
                        "representative={} size={} decks={}",
                        cluster.representative_id,
                        cluster.size(),
                        cluster.deck_names.join(",")
                    ))
                }));
                lines
            })
            .unwrap_or_else(|| vec![Line::from("Run duplicate analysis to inspect clusters.")]),
    }
}

fn workflow_form_lines(state: &WorkflowsState) -> Vec<Line<'static>> {
    match state.tab {
        WorkflowTab::Sync => vec![
            field_line(
                state.selected_field == WorkflowField::InputA,
                format!("source (.anki2): {}", empty_placeholder(&state.sync_source)),
            ),
            field_line(
                state.selected_field == WorkflowField::ToggleA,
                format!("run migrations: {}", checkbox(state.sync_run_migrations)),
            ),
            field_line(
                state.selected_field == WorkflowField::ToggleB,
                format!("run index: {}", checkbox(state.sync_run_index)),
            ),
            field_line(
                state.selected_field == WorkflowField::ToggleC,
                format!("force reindex: {}", checkbox(state.sync_force_reindex)),
            ),
            field_line(
                state.selected_field == WorkflowField::Run,
                "[ start sync ]".to_string(),
            ),
        ],
        WorkflowTab::Index => vec![
            field_line(
                state.selected_field == WorkflowField::ToggleA,
                format!("force reindex: {}", checkbox(state.index_force_reindex)),
            ),
            field_line(
                state.selected_field == WorkflowField::Run,
                "[ run index ]".to_string(),
            ),
        ],
        WorkflowTab::Generate => vec![
            field_line(
                state.selected_field == WorkflowField::InputA,
                format!("markdown file: {}", empty_placeholder(&state.generate_file)),
            ),
            field_line(
                state.selected_field == WorkflowField::Run,
                "[ preview cards ]".to_string(),
            ),
        ],
        WorkflowTab::Validate => vec![
            field_line(
                state.selected_field == WorkflowField::InputA,
                format!("cards file: {}", empty_placeholder(&state.validate_file)),
            ),
            field_line(
                state.selected_field == WorkflowField::ToggleA,
                format!("include quality: {}", checkbox(state.validate_quality)),
            ),
            field_line(
                state.selected_field == WorkflowField::Run,
                "[ validate file ]".to_string(),
            ),
        ],
        WorkflowTab::Obsidian => vec![
            field_line(
                state.selected_field == WorkflowField::InputA,
                format!("vault path: {}", empty_placeholder(&state.obsidian_vault)),
            ),
            field_line(
                state.selected_field == WorkflowField::InputB,
                format!(
                    "source dirs(csv): {}",
                    empty_placeholder(&state.obsidian_source_dirs)
                ),
            ),
            field_line(
                state.selected_field == WorkflowField::ToggleA,
                format!("dry run: {}", checkbox(state.obsidian_dry_run)),
            ),
            field_line(
                state.selected_field == WorkflowField::Run,
                "[ scan vault ]".to_string(),
            ),
        ],
        WorkflowTab::TagAudit => vec![
            field_line(
                state.selected_field == WorkflowField::InputA,
                format!("tag file: {}", empty_placeholder(&state.tag_audit_file)),
            ),
            field_line(
                state.selected_field == WorkflowField::ToggleA,
                format!("apply fixes: {}", checkbox(state.tag_audit_fix)),
            ),
            field_line(
                state.selected_field == WorkflowField::Run,
                "[ audit tags ]".to_string(),
            ),
        ],
    }
}

fn workflow_result_lines(state: &WorkflowsState) -> Vec<Line<'static>> {
    if let Some(error) = &state.error {
        return vec![Line::from(format!("error: {error}"))];
    }

    match state.tab {
        WorkflowTab::Sync => state
            .sync_result
            .as_ref()
            .map(|summary| {
                let mut lines = vec![
                    Line::from(format!("source: {}", summary.source.display())),
                    Line::from(format!(
                        "migrations_applied: {}",
                        summary.migrations_applied
                    )),
                    Line::from(format!("notes_upserted: {}", summary.sync.notes_upserted)),
                    Line::from(format!("notes_deleted: {}", summary.sync.notes_deleted)),
                    Line::from(format!("cards_upserted: {}", summary.sync.cards_upserted)),
                ];
                if let Some(index) = &summary.index {
                    lines.push(Line::from(format!(
                        "index embedded: {}",
                        index.stats.notes_embedded
                    )));
                }
                lines
            })
            .unwrap_or_else(|| vec![Line::from("Run sync to view the execution summary.")]),
        WorkflowTab::Index => state
            .index_result
            .as_ref()
            .map(|summary| {
                vec![
                    Line::from(format!("force_reindex: {}", summary.force_reindex)),
                    Line::from(format!(
                        "notes_processed: {}",
                        summary.stats.notes_processed
                    )),
                    Line::from(format!("notes_embedded: {}", summary.stats.notes_embedded)),
                    Line::from(format!("notes_skipped: {}", summary.stats.notes_skipped)),
                    Line::from(format!("notes_deleted: {}", summary.stats.notes_deleted)),
                ]
            })
            .unwrap_or_else(|| vec![Line::from("Run index to view the execution summary.")]),
        WorkflowTab::Generate => state
            .generate_result
            .as_ref()
            .map(|preview| {
                vec![
                    Line::from(format!("source: {}", preview.source_file.display())),
                    Line::from(format!(
                        "title: {}",
                        preview.title.as_deref().unwrap_or("(untitled)")
                    )),
                    Line::from(format!("estimated_cards: {}", preview.estimated_cards)),
                    Line::from(format!("sections: {}", preview.sections.join(", "))),
                ]
            })
            .unwrap_or_else(|| vec![Line::from("Run generate preview to inspect cards.")]),
        WorkflowTab::Validate => state
            .validate_result
            .as_ref()
            .map(|summary| {
                let mut lines = vec![
                    Line::from(format!("source: {}", summary.source_file.display())),
                    Line::from(format!("valid: {}", summary.is_valid)),
                    Line::from(format!("issues: {}", summary.issues.len())),
                ];
                lines.extend(summary.issues.iter().take(8).map(|issue| {
                    Line::from(format!(
                        "{} [{}] {}",
                        issue.severity, issue.location, issue.message
                    ))
                }));
                lines
            })
            .unwrap_or_else(|| vec![Line::from("Run validation to inspect issues.")]),
        WorkflowTab::Obsidian => state
            .obsidian_result
            .as_ref()
            .map(|preview| {
                vec![
                    Line::from(format!("vault: {}", preview.vault_path.display())),
                    Line::from(format!("notes: {}", preview.note_count)),
                    Line::from(format!("generated_cards: {}", preview.generated_cards)),
                    Line::from(format!("orphaned_notes: {}", preview.orphaned_notes.len())),
                    Line::from(format!("broken_links: {}", preview.broken_links.len())),
                ]
            })
            .unwrap_or_else(|| {
                vec![Line::from(
                    "Run obsidian scan to inspect the vault preview.",
                )]
            }),
        WorkflowTab::TagAudit => state
            .tag_audit_result
            .as_ref()
            .map(|summary| {
                let mut lines = vec![
                    Line::from(format!("source: {}", summary.source_file.display())),
                    Line::from(format!("applied_fixes: {}", summary.applied_fixes)),
                ];
                lines.extend(summary.entries.iter().take(10).map(|entry| {
                    Line::from(format!(
                        "{} valid={} normalized={}",
                        entry.tag, entry.valid, entry.normalized
                    ))
                }));
                lines
            })
            .unwrap_or_else(|| {
                vec![Line::from(
                    "Run tag audit to inspect normalization results.",
                )]
            }),
    }
}

fn field_line(selected: bool, content: String) -> Line<'static> {
    let style = if selected {
        Style::default()
            .fg(Color::Black)
            .bg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default()
    };
    Line::from(Span::styled(content, style))
}

fn display_field_value(value: &str, editing: bool) -> String {
    if editing {
        format!("{value}_")
    } else {
        empty_placeholder(value)
    }
}

fn empty_placeholder(value: &str) -> String {
    if value.trim().is_empty() {
        "(empty)".to_string()
    } else {
        value.to_string()
    }
}

fn checkbox(value: bool) -> &'static str {
    if value { "[x]" } else { "[ ]" }
}

fn region_border(focused: bool) -> Style {
    if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    }
}

fn split_csv(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

fn next_in<T: Copy + PartialEq, const N: usize>(items: [T; N], current: T) -> T {
    let index = items.iter().position(|item| *item == current).unwrap_or(0);
    items[(index + 1) % N]
}

fn previous_in<T: Copy + PartialEq, const N: usize>(items: [T; N], current: T) -> T {
    let index = items.iter().position(|item| *item == current).unwrap_or(0);
    if index == 0 {
        items[N - 1]
    } else {
        items[index - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use ratatui::backend::TestBackend;
    use search::fts::LexicalMode;
    use search::fusion::{FusionStats, SearchResult};
    use surface_runtime::SurfaceOperation;

    fn draw_app(app: &AppState) -> String {
        let backend = TestBackend::new(100, 32);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|frame| render(frame, app)).unwrap();
        format!("{:?}", terminal.backend())
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn sample_search_result() -> HybridSearchResult {
        HybridSearchResult {
            results: vec![SearchResult {
                note_id: 42,
                rrf_score: 0.9876,
                semantic_score: Some(0.91),
                semantic_rank: Some(1),
                fts_score: Some(0.73),
                fts_rank: Some(2),
                headline: Some("Rust ownership".to_string()),
                rerank_score: Some(0.88),
                rerank_rank: Some(1),
            }],
            stats: FusionStats {
                both: 1,
                total: 1,
                ..FusionStats::default()
            },
            query: "rust".to_string(),
            filters_applied: HashMap::new(),
            lexical_mode: LexicalMode::Fts,
            lexical_fallback_used: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
            rerank_applied: true,
            rerank_model: Some("reranker".to_string()),
            rerank_top_n: Some(5),
        }
    }

    fn sample_coverage() -> TopicCoverage {
        TopicCoverage {
            topic_id: 7,
            path: "rust/ownership".to_string(),
            label: "Ownership".to_string(),
            note_count: 12,
            subtree_count: 4,
            child_count: 3,
            covered_children: 2,
            mature_count: 5,
            avg_confidence: 0.83,
            weak_notes: 1,
            avg_lapses: 0.5,
        }
    }

    fn sample_generate_preview() -> GeneratePreview {
        GeneratePreview {
            source_file: "notes.md".into(),
            title: Some("Rust Ownership".to_string()),
            sections: vec!["Ownership".to_string(), "Borrowing".to_string()],
            estimated_cards: 3,
            warnings: vec![],
            cards: vec![],
        }
    }

    #[test]
    fn tab_cycles_search_fields() {
        let mut app = AppState {
            focus_region: FocusRegion::Content,
            screen: Screen::Search,
            ..Default::default()
        };
        handle_key_event(&mut app, key(KeyCode::Tab), &mpsc::unbounded_channel().0);
        assert_eq!(app.search.selected_field, SearchField::Decks);
    }

    #[test]
    fn help_modal_opens_and_closes() {
        let mut app = AppState::default();
        let tx = mpsc::unbounded_channel().0;
        handle_key_event(&mut app, key(KeyCode::Char('?')), &tx);
        assert_eq!(app.modal, Some(ModalState::Help));
        handle_key_event(&mut app, key(KeyCode::Esc), &tx);
        assert_eq!(app.modal, None);
    }

    #[test]
    fn slash_shortcut_focuses_search_query_editor() {
        let mut app = AppState::default();
        handle_key_event(
            &mut app,
            key(KeyCode::Char('/')),
            &mpsc::unbounded_channel().0,
        );

        assert_eq!(app.screen, Screen::Search);
        assert_eq!(app.focus_region, FocusRegion::Content);
        assert_eq!(app.search.selected_field, SearchField::Query);
        assert!(app.search.editing);
    }

    #[test]
    fn escape_from_content_returns_navigation_focus() {
        let app = AppState {
            screen: Screen::Search,
            focus_region: FocusRegion::Content,
            search: SearchState {
                editing: true,
                ..SearchState::default()
            },
            ..Default::default()
        };
        let mut app = app;

        handle_key_event(&mut app, key(KeyCode::Esc), &mpsc::unbounded_channel().0);

        assert_eq!(app.focus_region, FocusRegion::Navigation);
        assert!(!app.search.editing);
    }

    #[test]
    fn escape_from_navigation_returns_home_screen() {
        let mut app = AppState {
            screen: Screen::Workflows,
            navigation_index: 3,
            ..Default::default()
        };

        handle_key_event(&mut app, key(KeyCode::Esc), &mpsc::unbounded_channel().0);

        assert_eq!(app.screen, Screen::Home);
        assert_eq!(app.navigation_index, 0);
    }

    #[test]
    fn navigation_keys_switch_screens_and_enter_focuses_content() {
        let mut app = AppState::default();

        handle_key_event(&mut app, key(KeyCode::Down), &mpsc::unbounded_channel().0);
        assert_eq!(app.screen, Screen::Search);

        handle_key_event(&mut app, key(KeyCode::Down), &mpsc::unbounded_channel().0);
        assert_eq!(app.screen, Screen::Topics);

        handle_key_event(&mut app, key(KeyCode::Enter), &mpsc::unbounded_channel().0);
        assert_eq!(app.focus_region, FocusRegion::Content);
    }

    #[test]
    fn enter_while_editing_search_text_field_exits_edit_mode() {
        let mut app = AppState {
            screen: Screen::Search,
            focus_region: FocusRegion::Content,
            search: SearchState {
                selected_field: SearchField::Query,
                editing: true,
                query: "rust".to_string(),
                ..SearchState::default()
            },
            ..Default::default()
        };

        handle_key_event(&mut app, key(KeyCode::Enter), &mpsc::unbounded_channel().0);

        assert!(!app.search.editing);
        assert_eq!(app.search.query, "rust");
    }

    #[test]
    fn enter_while_editing_topics_text_field_exits_edit_mode() {
        let mut app = AppState {
            screen: Screen::Topics,
            focus_region: FocusRegion::Content,
            topics: TopicsState {
                tab: TopicsTab::Coverage,
                selected_field: TopicsField::InputA,
                editing: true,
                coverage_topic: "rust/ownership".to_string(),
                ..TopicsState::default()
            },
            ..Default::default()
        };

        handle_key_event(&mut app, key(KeyCode::Enter), &mpsc::unbounded_channel().0);

        assert!(!app.topics.editing);
        assert_eq!(app.topics.coverage_topic, "rust/ownership");
    }

    #[test]
    fn enter_while_editing_workflow_text_field_exits_edit_mode() {
        let mut app = AppState {
            screen: Screen::Workflows,
            focus_region: FocusRegion::Content,
            workflows: WorkflowsState {
                tab: WorkflowTab::Generate,
                selected_field: WorkflowField::InputA,
                editing: true,
                generate_file: "notes.md".to_string(),
                ..WorkflowsState::default()
            },
            ..Default::default()
        };

        handle_key_event(&mut app, key(KeyCode::Enter), &mpsc::unbounded_channel().0);

        assert!(!app.workflows.editing);
        assert_eq!(app.workflows.generate_file, "notes.md");
    }

    #[test]
    fn search_success_event_updates_result_summary_and_activity() {
        let mut app = AppState {
            screen: Screen::Search,
            ..Default::default()
        };

        app.apply_event(AppEvent::TaskFinished {
            label: "search".to_string(),
            result: Box::new(Ok(TaskResult::Search(sample_search_result()))),
        });

        assert_eq!(
            app.search
                .result
                .as_ref()
                .map(|result| result.results.len()),
            Some(1)
        );
        assert_eq!(app.search.selected_result, 0);
        assert_eq!(
            app.last_result_summary.as_deref(),
            Some("search returned 1 results")
        );
        assert_eq!(
            app.activity.front().map(std::string::String::as_str),
            Some("search finished")
        );
    }

    #[test]
    fn non_bootstrap_failure_event_is_routed_to_active_panel() {
        let mut app = AppState {
            screen: Screen::Topics,
            ..Default::default()
        };

        app.apply_event(AppEvent::TaskFinished {
            label: "coverage".to_string(),
            result: Box::new(Err("coverage exploded".to_string())),
        });

        assert_eq!(app.topics.error.as_deref(), Some("coverage exploded"));
        assert_eq!(app.last_result_summary.as_deref(), Some("coverage failed"));
        assert_eq!(
            app.activity.front().map(std::string::String::as_str),
            Some("coverage failed: coverage exploded")
        );
    }

    #[test]
    fn progress_event_updates_status_state() {
        let mut app = AppState::default();
        app.apply_event(AppEvent::Progress(SurfaceProgressEvent {
            operation: SurfaceOperation::Index,
            stage: "embedding".to_string(),
            current: 2,
            total: 5,
            message: "embedding notes".to_string(),
        }));

        assert_eq!(
            app.progress
                .as_ref()
                .map(|progress| progress.stage.as_str()),
            Some("embedding")
        );
        assert_eq!(
            app.progress
                .as_ref()
                .map(|progress| progress.message.as_str()),
            Some("embedding notes")
        );
    }

    #[test]
    fn activity_log_keeps_latest_entries_only() {
        let mut app = AppState::default();

        for index in 0..10 {
            app.push_activity(format!("event {index}"));
        }

        assert_eq!(app.activity.len(), 8);
        assert_eq!(
            app.activity.front().map(std::string::String::as_str),
            Some("event 9")
        );
        assert_eq!(
            app.activity.back().map(std::string::String::as_str),
            Some("event 2")
        );
    }

    #[test]
    fn render_home_screen_contains_bootstrap() {
        let app = AppState::default();
        let debug = draw_app(&app);
        assert!(debug.contains("Bootstrap"));
        assert!(debug.contains("Quick Actions"));
    }

    #[test]
    fn render_home_ready_screen_contains_runtime_summary() {
        let app = AppState {
            runtime: RuntimeState {
                status: RuntimeStatus::Ready,
                summary: Some(RuntimeSettingsSummary {
                    postgres_url: "postgres://db".to_string(),
                    qdrant_url: "http://qdrant".to_string(),
                    redis_url: "redis://cache".to_string(),
                    embedding_provider: "openai".to_string(),
                    embedding_model: "text-embedding-3-small".to_string(),
                    rerank_enabled: true,
                }),
                ..RuntimeState::default()
            },
            ..Default::default()
        };

        let debug = draw_app(&app);
        assert!(debug.contains("postgres://db"));
        assert!(debug.contains("text-embedding-3-small"));
        assert!(debug.contains("Rerank enabled: true"));
    }

    #[test]
    fn render_search_screen_contains_results_panel() {
        let app = AppState {
            screen: Screen::Search,
            ..Default::default()
        };
        let debug = draw_app(&app);
        assert!(debug.contains("Search Form"));
        assert!(debug.contains("Results"));
    }

    #[test]
    fn render_search_screen_contains_verbose_result_detail() {
        let app = AppState {
            screen: Screen::Search,
            search: SearchState {
                result: Some(sample_search_result()),
                verbose: true,
                ..SearchState::default()
            },
            ..Default::default()
        };

        let debug = draw_app(&app);
        assert!(debug.contains("Rust ownership"));
        assert!(debug.contains("semantic=Some(0.91)"));
    }

    #[test]
    fn render_topics_screen_contains_topic_result() {
        let app = AppState {
            screen: Screen::Topics,
            ..Default::default()
        };
        let debug = draw_app(&app);
        assert!(debug.contains("Topic Result"));
        assert!(debug.contains("Topic Views"));
    }

    #[test]
    fn render_topics_screen_contains_coverage_result_details() {
        let app = AppState {
            screen: Screen::Topics,
            topics: TopicsState {
                tab: TopicsTab::Coverage,
                coverage_result: Some(sample_coverage()),
                ..TopicsState::default()
            },
            ..Default::default()
        };

        let debug = draw_app(&app);
        assert!(debug.contains("rust/ownership"));
        assert!(debug.contains("covered_children: 2/3"));
    }

    #[test]
    fn render_workflows_screen_contains_workflow_result() {
        let app = AppState {
            screen: Screen::Workflows,
            ..Default::default()
        };
        let debug = draw_app(&app);
        assert!(debug.contains("Workflow Result"));
        assert!(debug.contains("Workflow Views"));
    }

    #[test]
    fn render_workflows_screen_contains_generate_preview() {
        let app = AppState {
            screen: Screen::Workflows,
            workflows: WorkflowsState {
                tab: WorkflowTab::Generate,
                generate_result: Some(sample_generate_preview()),
                ..WorkflowsState::default()
            },
            ..Default::default()
        };

        let debug = draw_app(&app);
        assert!(debug.contains("Rust Ownership"));
        assert!(debug.contains("estimated_cards: 3"));
    }
}

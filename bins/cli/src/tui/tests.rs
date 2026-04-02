use super::*;
use std::collections::HashMap;

use common::types::NoteId;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use surface_contracts::analytics::TopicCoverage;
use surface_contracts::search::{FusionStats, LexicalMode, SearchResponse, SearchResultItem};
use surface_runtime::{GeneratePreview, SurfaceOperation, SurfaceProgressEvent};
use tokio::sync::mpsc;

use crate::runtime::RuntimeSettingsSummary;

fn draw_app(app: &AppState) -> String {
    let backend = TestBackend::new(100, 32);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|frame| render(frame, app)).unwrap();
    format!("{:?}", terminal.backend())
}

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn sample_search_result() -> SearchResponse {
    SearchResponse {
        results: vec![SearchResultItem {
            note_id: NoteId(42),
            rrf_score: 0.9876,
            semantic_score: Some(0.91),
            semantic_rank: Some(1),
            fts_score: Some(0.73),
            fts_rank: Some(2),
            headline: Some("Rust ownership".to_string()),
            rerank_score: Some(0.88),
            rerank_rank: Some(1),
            sources: vec!["semantic".to_string(), "fts".to_string()],
            match_modality: Some("text".to_string()),
            match_chunk_kind: Some("text_primary".to_string()),
            match_source_field: None,
            match_asset_rel_path: None,
            match_preview_label: Some("Rust ownership".to_string()),
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

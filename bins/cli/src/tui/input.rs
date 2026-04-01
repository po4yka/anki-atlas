use crossterm::event::{KeyCode, KeyEvent};
use tokio::sync::mpsc;

use super::{
    AppEvent, AppState, FocusRegion, ModalState, RuntimeStatus, Screen, SearchField, TopicsField,
    TopicsTab, WorkflowField, WorkflowTab, run_topics_task, run_workflow_task, start_bootstrap,
};

pub(crate) fn handle_key_event(
    app: &mut AppState,
    key: KeyEvent,
    tx: &mpsc::UnboundedSender<AppEvent>,
) {
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
        if let FocusRegion::Content = app.focus_region {
            app.focus_region = FocusRegion::Navigation;
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
            if let Some(result) = &app.search.result
                && !result.results.is_empty()
            {
                app.search.selected_result =
                    (app.search.selected_result + 1).min(result.results.len() - 1);
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
                    super::tasks::spawn_task(tx.clone(), "search", move || async move {
                        crate::usecases::search(handles, request)
                            .await
                            .map(super::TaskResult::Search)
                            .map_err(|error| error.to_string())
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
            WorkflowField::InputA | WorkflowField::InputB => {
                app.workflows.editing = app.workflows.current_text_field_mut().is_some();
            }
            WorkflowField::InputC if app.workflows.tab == WorkflowTab::Obsidian => {
                app.workflows.editing = app.workflows.current_text_field_mut().is_some();
            }
            WorkflowField::ToggleA | WorkflowField::ToggleB | WorkflowField::ToggleC => {
                toggle_workflow_checkbox(&mut app.workflows);
            }
            WorkflowField::Run => run_workflow_task(app, tx),
            WorkflowField::InputC => {}
        },
        KeyCode::Char(' ') => match app.workflows.selected_field {
            WorkflowField::ToggleA | WorkflowField::ToggleB | WorkflowField::ToggleC => {
                toggle_workflow_checkbox(&mut app.workflows);
            }
            _ => {}
        },
        _ => {}
    }
}

fn handle_text_edit(target: Option<&mut String>, key: KeyEvent) -> bool {
    let Some(target) = target else {
        return false;
    };

    match key.code {
        KeyCode::Char(character) => {
            target.push(character);
            true
        }
        KeyCode::Backspace => {
            target.pop();
            true
        }
        _ => false,
    }
}

fn toggle_workflow_checkbox(workflows: &mut super::WorkflowsState) {
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

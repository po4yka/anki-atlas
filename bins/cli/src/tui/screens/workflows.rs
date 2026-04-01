use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs, Wrap};
use surface_runtime::{
    GeneratePreview, IndexExecutionSummary, ObsidianScanPreview, SyncExecutionSummary,
    TagAuditSummary, ValidationSummary,
};

use super::super::{AppState, checkbox, empty_placeholder, field_line, next_in, previous_in};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkflowTab {
    Sync,
    Index,
    Generate,
    Validate,
    Obsidian,
    TagAudit,
}

impl WorkflowTab {
    pub(crate) const ALL: [WorkflowTab; 6] = [
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

    pub(crate) fn next(self) -> Self {
        next_in(Self::ALL, self)
    }

    pub(crate) fn previous(self) -> Self {
        previous_in(Self::ALL, self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkflowField {
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
pub(crate) struct WorkflowsState {
    pub(crate) tab: WorkflowTab,
    pub(crate) selected_field: WorkflowField,
    pub(crate) editing: bool,
    pub(crate) sync_source: String,
    pub(crate) sync_run_migrations: bool,
    pub(crate) sync_run_index: bool,
    pub(crate) sync_force_reindex: bool,
    pub(crate) index_force_reindex: bool,
    pub(crate) generate_file: String,
    pub(crate) validate_file: String,
    pub(crate) validate_quality: bool,
    pub(crate) obsidian_vault: String,
    pub(crate) obsidian_source_dirs: String,
    pub(crate) obsidian_dry_run: bool,
    pub(crate) tag_audit_file: String,
    pub(crate) tag_audit_fix: bool,
    pub(crate) sync_result: Option<SyncExecutionSummary>,
    pub(crate) index_result: Option<IndexExecutionSummary>,
    pub(crate) generate_result: Option<GeneratePreview>,
    pub(crate) validate_result: Option<ValidationSummary>,
    pub(crate) obsidian_result: Option<ObsidianScanPreview>,
    pub(crate) tag_audit_result: Option<TagAuditSummary>,
    pub(crate) error: Option<String>,
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
    pub(crate) fn next_field(&mut self) {
        self.selected_field = self.selected_field.next();
    }

    pub(crate) fn previous_field(&mut self) {
        self.selected_field = self.selected_field.previous();
    }

    pub(crate) fn current_text_field_mut(&mut self) -> Option<&mut String> {
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

pub(crate) fn render_workflows(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
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

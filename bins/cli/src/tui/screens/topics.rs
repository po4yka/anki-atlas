use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs, Wrap};
use surface_contracts::analytics::{
    DuplicateCluster, DuplicateStats, TopicCoverage, TopicGap, WeakNote,
};

use super::super::{AppState, checkbox, empty_placeholder, field_line, next_in, previous_in};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TopicsTab {
    Tree,
    Coverage,
    Gaps,
    WeakNotes,
    Duplicates,
}

impl TopicsTab {
    pub(crate) const ALL: [TopicsTab; 5] = [
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

    pub(crate) fn next(self) -> Self {
        next_in(Self::ALL, self)
    }

    pub(crate) fn previous(self) -> Self {
        previous_in(Self::ALL, self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TopicsField {
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
pub(crate) struct TopicsState {
    pub(crate) tab: TopicsTab,
    pub(crate) selected_field: TopicsField,
    pub(crate) editing: bool,
    pub(crate) tree_root: String,
    pub(crate) coverage_topic: String,
    pub(crate) coverage_include_subtree: bool,
    pub(crate) gaps_topic: String,
    pub(crate) gaps_min_coverage: String,
    pub(crate) weak_topic: String,
    pub(crate) weak_limit: String,
    pub(crate) duplicates_threshold: String,
    pub(crate) duplicates_max: String,
    pub(crate) duplicates_decks: String,
    pub(crate) duplicates_tags: String,
    pub(crate) tree_result: Option<Vec<serde_json::Value>>,
    pub(crate) coverage_result: Option<TopicCoverage>,
    pub(crate) gaps_result: Option<Vec<TopicGap>>,
    pub(crate) weak_notes_result: Option<Vec<WeakNote>>,
    pub(crate) duplicates_result: Option<(Vec<DuplicateCluster>, DuplicateStats)>,
    pub(crate) error: Option<String>,
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
    pub(crate) fn next_field(&mut self) {
        self.selected_field = self.selected_field.next();
    }

    pub(crate) fn previous_field(&mut self) {
        self.selected_field = self.selected_field.previous();
    }

    pub(crate) fn current_text_field_mut(&mut self) -> Option<&mut String> {
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

pub(crate) fn render_topics(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
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

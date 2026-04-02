use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use surface_contracts::search::{SearchMode, SearchResponse};

use crate::usecases::SearchRequest;

use super::super::{
    AppState, FocusRegion, checkbox, display_field_value, empty_placeholder, field_line, next_in,
    previous_in, split_csv,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SearchField {
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
pub(crate) struct SearchState {
    pub(crate) query: String,
    pub(crate) deck_names: String,
    pub(crate) tags: String,
    pub(crate) limit: String,
    pub(crate) search_mode: SearchMode,
    pub(crate) verbose: bool,
    pub(crate) selected_field: SearchField,
    pub(crate) editing: bool,
    pub(crate) result: Option<SearchResponse>,
    pub(crate) error: Option<String>,
    pub(crate) selected_result: usize,
}

impl Default for SearchState {
    fn default() -> Self {
        Self {
            query: String::new(),
            deck_names: String::new(),
            tags: String::new(),
            limit: "10".to_string(),
            search_mode: SearchMode::Hybrid,
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
    pub(crate) fn current_text_field_mut(&mut self) -> Option<&mut String> {
        match self.selected_field {
            SearchField::Query => Some(&mut self.query),
            SearchField::Decks => Some(&mut self.deck_names),
            SearchField::Tags => Some(&mut self.tags),
            SearchField::Limit => Some(&mut self.limit),
            _ => None,
        }
    }

    pub(crate) fn next_field(&mut self) {
        self.selected_field = self.selected_field.next();
    }

    pub(crate) fn previous_field(&mut self) {
        self.selected_field = self.selected_field.previous();
    }

    pub(crate) fn request(&self) -> SearchRequest {
        SearchRequest {
            query: self.query.clone(),
            deck_names: split_csv(&self.deck_names),
            tags: split_csv(&self.tags),
            limit: self.limit.parse::<usize>().unwrap_or(10).max(1),
            search_mode: self.search_mode,
        }
    }

    pub(crate) fn toggle_semantic(&mut self) {
        self.search_mode = match self.search_mode {
            SearchMode::SemanticOnly => SearchMode::Hybrid,
            _ => SearchMode::SemanticOnly,
        };
    }

    pub(crate) fn toggle_fts(&mut self) {
        self.search_mode = match self.search_mode {
            SearchMode::FtsOnly => SearchMode::Hybrid,
            _ => SearchMode::FtsOnly,
        };
    }
}

pub(crate) fn render_search(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
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
            format!(
                "semantic only: {}",
                checkbox(state.search_mode == SearchMode::SemanticOnly)
            ),
        ),
        field_line(
            state.selected_field == SearchField::Fts && content_focused,
            format!(
                "fts only: {}",
                checkbox(state.search_mode == SearchMode::FtsOnly)
            ),
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
        Line::from(format!("sources: {}", item.sources.join(","))),
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

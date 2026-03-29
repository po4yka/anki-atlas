use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Wrap};

use super::{
    AppState, FocusRegion, Screen, render_home, render_search, render_topics, render_workflows,
};

pub(crate) fn render(frame: &mut Frame<'_>, app: &AppState) {
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

pub(crate) fn field_line(selected: bool, content: String) -> Line<'static> {
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

pub(crate) fn display_field_value(value: &str, editing: bool) -> String {
    if editing {
        format!("{value}_")
    } else {
        empty_placeholder(value)
    }
}

pub(crate) fn empty_placeholder(value: &str) -> String {
    if value.trim().is_empty() {
        "(empty)".to_string()
    } else {
        value.to_string()
    }
}

pub(crate) fn checkbox(value: bool) -> &'static str {
    if value { "[x]" } else { "[ ]" }
}

pub(crate) fn region_border(focused: bool) -> Style {
    if focused {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default()
    }
}

pub(crate) fn split_csv(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToString::to_string)
        .collect()
}

pub(crate) fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
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

pub(crate) fn next_in<T: Copy + PartialEq, const N: usize>(items: [T; N], current: T) -> T {
    let index = items.iter().position(|item| *item == current).unwrap_or(0);
    items[(index + 1) % N]
}

pub(crate) fn previous_in<T: Copy + PartialEq, const N: usize>(items: [T; N], current: T) -> T {
    let index = items.iter().position(|item| *item == current).unwrap_or(0);
    if index == 0 {
        items[N - 1]
    } else {
        items[index - 1]
    }
}

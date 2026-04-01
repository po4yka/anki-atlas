use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use super::super::{AppState, RuntimeStatus};

pub(crate) fn render_home(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
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

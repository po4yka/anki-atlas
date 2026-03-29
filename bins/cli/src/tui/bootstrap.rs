use std::io::{self, Stdout};
use std::time::Duration;

use crossterm::event::{self, Event};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Frame;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;

use crate::runtime::{ExecutionMode, bootstrap_runtime};

use super::{AppEvent, AppState, TaskResult, handle_key_event, render};

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

        if event::poll(Duration::from_millis(50))?
            && let Event::Key(key) = event::read()?
        {
            handle_key_event(&mut app, key, &tx);
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

pub(crate) fn start_bootstrap(
    tx: mpsc::UnboundedSender<AppEvent>,
    enable_direct_execution: bool,
) {
    tokio::spawn(async move {
        let mode = if enable_direct_execution {
            ExecutionMode::DirectExecution
        } else {
            ExecutionMode::ReadOnly
        };
        let event = match bootstrap_runtime(mode).await {
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

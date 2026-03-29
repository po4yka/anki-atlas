mod app;
mod bootstrap;
mod input;
mod screens;
mod tasks;
#[cfg(test)]
mod tests;
mod widgets;

pub use bootstrap::run;

pub(crate) use app::{
    AppEvent, AppState, FocusRegion, ModalState, RuntimeStatus, Screen, TaskResult,
};
#[cfg(test)]
pub(crate) use app::RuntimeState;
pub(crate) use bootstrap::start_bootstrap;
pub(crate) use input::handle_key_event;
pub(crate) use screens::home::render_home;
pub(crate) use screens::search::{SearchField, SearchState, render_search};
pub(crate) use screens::topics::{TopicsField, TopicsState, TopicsTab, render_topics};
pub(crate) use screens::workflows::{
    WorkflowField, WorkflowTab, WorkflowsState, render_workflows,
};
pub(crate) use tasks::{run_topics_task, run_workflow_task};
pub(crate) use widgets::{
    checkbox, display_field_value, empty_placeholder, field_line, next_in, previous_in, render,
    split_csv,
};

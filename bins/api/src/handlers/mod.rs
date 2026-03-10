mod duplicates;
mod health;
mod index_info;
mod jobs;
mod search;
mod sync;
mod topics;

use std::collections::HashMap;

use crate::error::AppError;

pub use duplicates::find_duplicates;
pub use health::{health, ready};
pub use index_info::index_info;
pub use jobs::{cancel_job, enqueue_index_job, enqueue_sync_job, get_job_status};
pub use search::search;
pub use sync::{index_notes, sync};
pub use topics::{list_topics, topic_wildcard};

pub(crate) fn unwired_surface(surface: &str, guidance: &str) -> AppError {
    AppError(
        common::error::AnkiAtlasError::Conflict {
            message: format!("{surface} is not wired to a domain service yet; {guidance}"),
            context: HashMap::new(),
        }
        .into(),
    )
}

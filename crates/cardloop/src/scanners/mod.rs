pub mod audit;
pub mod fsrs;
pub mod generation;
pub mod llm_review;

use crate::error::CardloopError;
use crate::models::WorkItem;

/// Trait for scanners that detect work items.
///
/// Scanners run single-threaded in the CLI, so no Send + Sync bound.
pub trait Scanner {
    /// Scan the data source and return new or updated work items.
    fn scan(&self, scan_number: u32) -> Result<Vec<WorkItem>, CardloopError>;
}

/// Trait for async scanners that detect work items.
#[cfg_attr(test, mockall::automock)]
#[async_trait::async_trait]
pub trait AsyncScanner: Send + Sync {
    /// Scan the data source and return new or updated work items.
    async fn scan(&self, scan_number: u32) -> Result<Vec<WorkItem>, CardloopError>;
}

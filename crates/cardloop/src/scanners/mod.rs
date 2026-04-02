pub mod audit;
pub mod generation;

use crate::error::CardloopError;
use crate::models::WorkItem;

/// Trait for scanners that detect work items.
///
/// Scanners run single-threaded in the CLI, so no Send + Sync bound.
pub trait Scanner {
    /// Scan the data source and return new or updated work items.
    fn scan(&self, scan_number: u32) -> Result<Vec<WorkItem>, CardloopError>;
}

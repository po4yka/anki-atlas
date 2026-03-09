use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Serialized job message pushed to the Redis queue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobEnvelope {
    pub job_id: String,
    pub task_name: String,
    pub payload: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests;

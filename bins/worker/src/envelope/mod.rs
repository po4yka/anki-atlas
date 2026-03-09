use jobs::types::JobType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Serialized job message pushed to the Redis queue.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobEnvelope {
    pub job_id: String,
    pub job_type: JobType,
    pub payload: HashMap<String, serde_json::Value>,
}

#[cfg(test)]
mod tests;

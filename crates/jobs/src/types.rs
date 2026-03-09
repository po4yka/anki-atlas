use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum::{Display, EnumString};

/// Job type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, EnumString, Display)]
#[strum(serialize_all = "lowercase")]
pub enum JobType {
    Sync,
    Index,
}

/// Job status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, EnumString, Display)]
#[strum(serialize_all = "lowercase")]
pub enum JobStatus {
    Queued,
    Scheduled,
    Running,
    Retrying,
    Succeeded,
    Failed,
    CancelRequested,
    Cancelled,
}

impl JobStatus {
    /// True if the job has reached a final state.
    pub fn is_terminal(&self) -> bool {
        // TODO(impl): implement
        false
    }
}

pub const JOB_KEY_PREFIX: &str = "ankiatlas:job:";

/// Persisted metadata for an async job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub job_id: String,
    pub job_type: JobType,
    pub status: JobStatus,
    pub payload: HashMap<String, serde_json::Value>,
    pub progress: f64,
    pub message: Option<String>,
    pub attempts: u32,
    pub max_retries: u32,
    pub cancel_requested: bool,
    pub created_at: Option<DateTime<Utc>>,
    pub scheduled_for: Option<DateTime<Utc>>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub result: Option<HashMap<String, serde_json::Value>>,
    pub error: Option<String>,
}

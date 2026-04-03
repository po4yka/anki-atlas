use chrono::{DateTime, Utc};
use common::ReindexMode;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

/// Job type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, EnumString, Display)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum JobType {
    Sync,
    Index,
}

/// Job status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, EnumString, Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
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
        matches!(self, Self::Succeeded | Self::Failed | Self::Cancelled)
    }
}

pub const JOB_KEY_PREFIX: &str = "ankiatlas:job:";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncJobPayload {
    pub source: String,
    pub run_migrations: bool,
    pub index: bool,
    pub reindex_mode: ReindexMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexJobPayload {
    pub reindex_mode: ReindexMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum JobPayload {
    Sync(SyncJobPayload),
    Index(IndexJobPayload),
}

impl JobPayload {
    pub fn job_type(&self) -> JobType {
        match self {
            Self::Sync(_) => JobType::Sync,
            Self::Index(_) => JobType::Index,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SyncJobResult {
    pub decks_upserted: i64,
    pub models_upserted: i64,
    pub notes_upserted: i64,
    pub notes_deleted: i64,
    pub cards_upserted: i64,
    pub card_stats_upserted: i64,
    pub duration_ms: i64,
    pub notes_embedded: Option<i64>,
    pub notes_skipped: Option<i64>,
    pub index_errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexJobResult {
    pub notes_processed: i64,
    pub notes_embedded: i64,
    pub notes_skipped: i64,
    pub notes_deleted: i64,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum JobResultData {
    Sync(SyncJobResult),
    Index(IndexJobResult),
}

/// Persisted metadata for an async job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub job_id: String,
    pub job_type: JobType,
    pub status: JobStatus,
    pub payload: JobPayload,
    pub progress: f64,
    pub message: Option<String>,
    pub attempts: u32,
    pub max_retries: u32,
    pub cancel_requested: bool,
    pub created_at: Option<DateTime<Utc>>,
    pub scheduled_for: Option<DateTime<Utc>>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub result: Option<JobResultData>,
    pub error: Option<String>,
}

/// Serialized job message pushed to the Redis queue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobEnvelope {
    pub job_id: String,
    pub job_type: JobType,
    pub payload: JobPayload,
}

impl From<&JobRecord> for JobEnvelope {
    fn from(record: &JobRecord) -> Self {
        Self {
            job_id: record.job_id.clone(),
            job_type: record.job_type,
            payload: record.payload.clone(),
        }
    }
}

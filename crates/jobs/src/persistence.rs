use sqlx::PgPool;

use crate::error::JobError;
use crate::types::JobRecord;

/// Ensure the job_queue table exists (idempotent).
pub async fn ensure_schema(pool: &PgPool) -> Result<(), JobError> {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS job_queue (
            job_id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            payload_json JSONB NOT NULL,
            progress DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            message TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 3,
            cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
            result_json JSONB,
            error TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            scheduled_for TIMESTAMPTZ,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            ttl_seconds INTEGER NOT NULL DEFAULT 86400
        )",
    )
    .execute(pool)
    .await?;

    // Index for queue polling (SKIP LOCKED pattern)
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_job_queue_pending
         ON job_queue (created_at)
         WHERE status = 'queued'",
    )
    .execute(pool)
    .await?;

    // Cleanup index for expired jobs
    sqlx::query(
        "CREATE INDEX IF NOT EXISTS idx_job_queue_cleanup
         ON job_queue (created_at, ttl_seconds)
         WHERE status IN ('succeeded', 'failed', 'cancelled')",
    )
    .execute(pool)
    .await?;

    Ok(())
}

/// Persist a job record (upsert).
pub async fn save_job_record(
    pool: &PgPool,
    record: &JobRecord,
    ttl_seconds: u64,
) -> Result<(), JobError> {
    let payload_json = serde_json::to_value(&record.payload)
        .map_err(|e| JobError::Serialization(e.to_string()))?;
    let result_json = record
        .result
        .as_ref()
        .map(|r| serde_json::to_value(r).unwrap_or_default());

    sqlx::query(
        "INSERT INTO job_queue (
            job_id, job_type, status, payload_json, progress, message,
            attempts, max_retries, cancel_requested, result_json, error,
            created_at, scheduled_for, started_at, finished_at, ttl_seconds
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        ON CONFLICT (job_id) DO UPDATE SET
            status = EXCLUDED.status,
            progress = EXCLUDED.progress,
            message = EXCLUDED.message,
            attempts = EXCLUDED.attempts,
            cancel_requested = EXCLUDED.cancel_requested,
            result_json = EXCLUDED.result_json,
            error = EXCLUDED.error,
            started_at = EXCLUDED.started_at,
            finished_at = EXCLUDED.finished_at",
    )
    .bind(&record.job_id)
    .bind(record.job_type.to_string())
    .bind(record.status.to_string())
    .bind(&payload_json)
    .bind(record.progress)
    .bind(&record.message)
    .bind(record.attempts as i32)
    .bind(record.max_retries as i32)
    .bind(record.cancel_requested)
    .bind(&result_json)
    .bind(&record.error)
    .bind(record.created_at)
    .bind(record.scheduled_for)
    .bind(record.started_at)
    .bind(record.finished_at)
    .bind(ttl_seconds as i32)
    .execute(pool)
    .await?;

    Ok(())
}

/// Load a job record by ID.
pub async fn load_job_record(pool: &PgPool, job_id: &str) -> Result<Option<JobRecord>, JobError> {
    let row = sqlx::query_as::<_, JobRow>(
        "SELECT job_id, job_type, status, payload_json, progress, message,
                attempts, max_retries, cancel_requested, result_json, error,
                created_at, scheduled_for, started_at, finished_at
         FROM job_queue WHERE job_id = $1",
    )
    .bind(job_id)
    .fetch_optional(pool)
    .await?;

    match row {
        Some(r) => Ok(Some(r.into_record()?)),
        None => Ok(None),
    }
}

/// Pop the next queued job (FIFO, using FOR UPDATE SKIP LOCKED).
///
/// Atomically selects the oldest queued job and sets its status to 'running'.
/// Returns None if no jobs are queued.
pub async fn pop_next_job(pool: &PgPool) -> Result<Option<JobRecord>, JobError> {
    let row = sqlx::query_as::<_, JobRow>(
        "UPDATE job_queue SET status = 'running', started_at = NOW(), attempts = attempts + 1
         WHERE job_id = (
            SELECT job_id FROM job_queue
            WHERE status = 'queued'
            ORDER BY created_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
         )
         RETURNING job_id, job_type, status, payload_json, progress, message,
                   attempts, max_retries, cancel_requested, result_json, error,
                   created_at, scheduled_for, started_at, finished_at",
    )
    .fetch_optional(pool)
    .await?;

    match row {
        Some(r) => Ok(Some(r.into_record()?)),
        None => Ok(None),
    }
}

/// Re-enqueue a job for retry (set status back to 'queued').
pub async fn reenqueue_job(pool: &PgPool, record: &JobRecord) -> Result<(), JobError> {
    sqlx::query(
        "UPDATE job_queue SET status = 'queued', started_at = NULL,
                attempts = $2, error = $3, message = $4
         WHERE job_id = $1",
    )
    .bind(&record.job_id)
    .bind(record.attempts as i32)
    .bind(&record.error)
    .bind(&record.message)
    .execute(pool)
    .await?;
    Ok(())
}

/// Delete expired terminal jobs.
pub async fn cleanup_expired(pool: &PgPool) -> Result<u64, JobError> {
    let result = sqlx::query(
        "DELETE FROM job_queue
         WHERE status IN ('succeeded', 'failed', 'cancelled')
         AND created_at + (ttl_seconds || ' seconds')::interval < NOW()",
    )
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

/// Internal row type for sqlx mapping.
#[derive(sqlx::FromRow)]
struct JobRow {
    job_id: String,
    job_type: String,
    status: String,
    payload_json: serde_json::Value,
    progress: f64,
    message: Option<String>,
    attempts: i32,
    max_retries: i32,
    cancel_requested: bool,
    result_json: Option<serde_json::Value>,
    error: Option<String>,
    created_at: Option<chrono::DateTime<chrono::Utc>>,
    scheduled_for: Option<chrono::DateTime<chrono::Utc>>,
    started_at: Option<chrono::DateTime<chrono::Utc>>,
    finished_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl JobRow {
    fn into_record(self) -> Result<JobRecord, JobError> {
        use std::str::FromStr;

        let job_type = crate::types::JobType::from_str(&self.job_type)
            .map_err(|_| JobError::Serialization(format!("unknown job_type: {}", self.job_type)))?;
        let status = crate::types::JobStatus::from_str(&self.status)
            .map_err(|_| JobError::Serialization(format!("unknown status: {}", self.status)))?;
        let payload = serde_json::from_value(self.payload_json)
            .map_err(|e| JobError::Serialization(format!("payload: {e}")))?;
        let result = self
            .result_json
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| JobError::Serialization(format!("result: {e}")))?;

        Ok(JobRecord {
            job_id: self.job_id,
            job_type,
            status,
            payload,
            progress: self.progress,
            message: self.message,
            attempts: self.attempts as u32,
            max_retries: self.max_retries as u32,
            cancel_requested: self.cancel_requested,
            created_at: self.created_at,
            scheduled_for: self.scheduled_for,
            started_at: self.started_at,
            finished_at: self.finished_at,
            result,
            error: self.error,
        })
    }
}

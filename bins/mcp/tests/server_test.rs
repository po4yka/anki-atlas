use std::path::PathBuf;
use std::sync::Arc;

use analytics::AnalyticsError;
use analytics::coverage::{TopicCoverage, TopicGap, WeakNote};
use analytics::duplicates::{DuplicateCluster, DuplicateStats};
use analytics::labeling::LabelingStats;
use analytics::taxonomy::Taxonomy;
use anki_atlas_mcp::server::AnkiAtlasServer;
use jobs::{IndexJobPayload, JobError, JobManager, JobRecord, SyncJobPayload};
use search::error::SearchError;
use search::service::{ChunkSearchParams, ChunkSearchResult, HybridSearchResult, SearchParams};
use surface_runtime::{AnalyticsFacade, SearchFacade, SurfaceServices};

struct NoopJobs;

#[async_trait::async_trait]
impl JobManager for NoopJobs {
    async fn enqueue_sync_job(
        &self,
        _payload: SyncJobPayload,
        _run_at: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<JobRecord, JobError> {
        Err(JobError::Unsupported(
            "not used in registration tests".to_string(),
        ))
    }

    async fn enqueue_index_job(
        &self,
        _payload: IndexJobPayload,
        _run_at: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<JobRecord, JobError> {
        Err(JobError::Unsupported(
            "not used in registration tests".to_string(),
        ))
    }

    async fn get_job(&self, _job_id: &str) -> Result<JobRecord, JobError> {
        Err(JobError::Unsupported(
            "not used in registration tests".to_string(),
        ))
    }

    async fn cancel_job(&self, _job_id: &str) -> Result<JobRecord, JobError> {
        Err(JobError::Unsupported(
            "not used in registration tests".to_string(),
        ))
    }

    async fn close(&self) -> Result<(), JobError> {
        Ok(())
    }
}

struct NoopSearch;

#[async_trait::async_trait]
impl SearchFacade for NoopSearch {
    async fn search(&self, _params: &SearchParams) -> Result<HybridSearchResult, SearchError> {
        Err(SearchError::Database(sqlx::Error::PoolTimedOut))
    }

    async fn search_chunks(
        &self,
        _params: &ChunkSearchParams,
    ) -> Result<ChunkSearchResult, SearchError> {
        Err(SearchError::Database(sqlx::Error::PoolTimedOut))
    }
}

struct NoopAnalytics;

#[async_trait::async_trait]
impl AnalyticsFacade for NoopAnalytics {
    async fn load_taxonomy(&self, _yaml_path: Option<PathBuf>) -> Result<Taxonomy, AnalyticsError> {
        Ok(Taxonomy::default())
    }

    async fn label_notes(
        &self,
        _yaml_path: Option<PathBuf>,
        _min_confidence: f32,
    ) -> Result<LabelingStats, AnalyticsError> {
        Ok(LabelingStats::default())
    }

    async fn get_taxonomy_tree(
        &self,
        _root_path: Option<String>,
    ) -> Result<Vec<serde_json::Value>, AnalyticsError> {
        Ok(Vec::new())
    }

    async fn get_coverage(
        &self,
        _topic_path: String,
        _include_subtree: bool,
    ) -> Result<Option<TopicCoverage>, AnalyticsError> {
        Ok(None)
    }

    async fn get_gaps(
        &self,
        _topic_path: String,
        _min_coverage: i64,
    ) -> Result<Vec<TopicGap>, AnalyticsError> {
        Ok(Vec::new())
    }

    async fn get_weak_notes(
        &self,
        _topic_path: String,
        _max_results: i64,
    ) -> Result<Vec<WeakNote>, AnalyticsError> {
        Ok(Vec::new())
    }

    async fn find_duplicates(
        &self,
        _threshold: f64,
        _max_clusters: usize,
        _deck_filter: Option<Vec<String>>,
        _tag_filter: Option<Vec<String>>,
    ) -> Result<(Vec<DuplicateCluster>, DuplicateStats), AnalyticsError> {
        Ok((Vec::new(), DuplicateStats::default()))
    }
}

fn test_server() -> AnkiAtlasServer {
    let pool = sqlx::postgres::PgPoolOptions::new()
        .connect_lazy("postgresql://localhost:5432/test")
        .expect("lazy pool");
    let services = SurfaceServices::new(
        pool,
        Arc::new(NoopJobs),
        Arc::new(NoopSearch),
        Arc::new(NoopAnalytics),
    );
    AnkiAtlasServer::new(Arc::new(services))
}

#[tokio::test]
async fn server_name_and_version_are_set() {
    let server = test_server();
    assert_eq!(server.name(), "anki-atlas");
    assert!(!server.version().is_empty());
}

#[tokio::test]
async fn server_registers_expected_tool_set() {
    let server = test_server();
    let names = server.tool_names();
    assert_eq!(server.tool_count(), 15);
    assert_eq!(
        names,
        vec![
            "ankiatlas_duplicates",
            "ankiatlas_generate",
            "ankiatlas_index_job",
            "ankiatlas_job_cancel",
            "ankiatlas_job_status",
            "ankiatlas_obsidian_sync",
            "ankiatlas_search",
            "ankiatlas_search_chunks",
            "ankiatlas_sync_job",
            "ankiatlas_tag_audit",
            "ankiatlas_topic_coverage",
            "ankiatlas_topic_gaps",
            "ankiatlas_topic_weak_notes",
            "ankiatlas_topics",
            "ankiatlas_validate",
        ]
    );
}

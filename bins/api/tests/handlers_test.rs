use anki_atlas_api::router::build_router;
use anki_atlas_api::state::AppState;
use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use common::config::Settings;
use jobs::{
    IndexJobPayload, JobError, JobManager, JobPayload, JobRecord, JobStatus, JobType,
    SyncJobPayload,
};
use mockall::mock;
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;

// Mock JobManager for tests
mock! {
    pub Jobs {}

    #[async_trait]
    impl JobManager for Jobs {
        async fn enqueue_sync_job(
            &self,
            payload: SyncJobPayload,
            run_at: Option<chrono::DateTime<chrono::Utc>>,
        ) -> Result<JobRecord, JobError>;

        async fn enqueue_index_job(
            &self,
            payload: IndexJobPayload,
            run_at: Option<chrono::DateTime<chrono::Utc>>,
        ) -> Result<JobRecord, JobError>;

        async fn get_job(&self, job_id: &str) -> Result<JobRecord, JobError>;

        async fn cancel_job(&self, job_id: &str) -> Result<JobRecord, JobError>;

        async fn close(&self) -> Result<(), JobError>;
    }
}

fn test_settings() -> Settings {
    Settings {
        postgres_url: "postgresql://localhost:5432/test".into(),
        qdrant_url: "http://localhost:6333".into(),
        qdrant_quantization: common::config::Quantization::None,
        qdrant_on_disk: false,
        redis_url: "redis://localhost:6379/0".into(),
        job_queue_name: "test_jobs".into(),
        job_result_ttl_seconds: 3600,
        job_max_retries: 3,
        embedding_provider: common::config::EmbeddingProviderKind::Mock,
        embedding_model: "test-model".into(),
        embedding_dimension: 384,
        rerank_enabled: false,
        rerank_model: "test-rerank".into(),
        rerank_top_n: 50,
        rerank_batch_size: 32,
        api_host: "0.0.0.0".into(),
        api_port: 8000,
        api_key: None,
        debug: false,
        anki_collection_path: None,
    }
}

fn test_state(mock_jobs: MockJobs) -> AppState {
    AppState {
        api: Arc::new(test_settings().api()),
        job_manager: Arc::new(mock_jobs),
    }
}

fn make_job_record(job_id: &str, job_type: JobType, status: JobStatus) -> JobRecord {
    let payload = match job_type {
        JobType::Sync => JobPayload::Sync(SyncJobPayload {
            source: "/tmp/collection.anki2".to_string(),
            run_migrations: true,
            index: true,
            force_reindex: false,
        }),
        JobType::Index => JobPayload::Index(IndexJobPayload {
            force_reindex: false,
        }),
    };

    JobRecord {
        job_id: job_id.to_string(),
        job_type,
        status,
        payload,
        progress: 0.0,
        message: None,
        attempts: 0,
        max_retries: 3,
        cancel_requested: false,
        created_at: Some(chrono::Utc::now()),
        scheduled_for: None,
        started_at: None,
        finished_at: None,
        result: None,
        error: None,
    }
}

async fn response_json(resp: Response) -> Value {
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}

// ---- Health ----

#[tokio::test]
async fn health_returns_200_with_status() {
    let app = build_router(test_state(MockJobs::new()));
    let resp: Response = app
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = response_json(resp).await;
    assert_eq!(v["status"], "healthy");
}

#[tokio::test]
async fn health_includes_version() {
    let app = build_router(test_state(MockJobs::new()));
    let resp: Response = app
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let v = response_json(resp).await;
    assert!(v["version"].is_string(), "health should include version");
}

// ---- Jobs ----

#[tokio::test]
async fn enqueue_sync_job_returns_202() {
    let mut mock = MockJobs::new();
    mock.expect_enqueue_sync_job()
        .returning(|_, _| Ok(make_job_record("job-1", JobType::Sync, JobStatus::Queued)));

    let app = build_router(test_state(mock));
    let body = json!({ "source": "/path/col.anki2" });
    let resp: Response = app
        .oneshot(
            Request::post("/jobs/sync")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
    let v = response_json(resp).await;
    assert_eq!(v["job_id"], "job-1");
    assert!(v["poll_url"].as_str().unwrap().contains("job-1"));
}

#[tokio::test]
async fn enqueue_index_job_returns_202() {
    let mut mock = MockJobs::new();
    mock.expect_enqueue_index_job()
        .returning(|_, _| Ok(make_job_record("job-2", JobType::Index, JobStatus::Queued)));

    let app = build_router(test_state(mock));
    let body = json!({});
    let resp: Response = app
        .oneshot(
            Request::post("/jobs/index")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
}

#[tokio::test]
async fn enqueue_sync_job_rejects_scheduling_until_supported() {
    let mut mock = MockJobs::new();
    mock.expect_enqueue_sync_job().returning(|_, _| {
        Err(JobError::Unsupported(
            "scheduled jobs are not supported yet".into(),
        ))
    });

    let app = build_router(test_state(mock));
    let body = json!({
        "source": "/path/col.anki2",
        "run_at": "2026-03-11T10:00:00Z"
    });
    let resp: Response = app
        .oneshot(
            Request::post("/jobs/sync")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn get_job_returns_404_for_unknown() {
    let mut mock = MockJobs::new();
    mock.expect_get_job()
        .returning(|job_id| Err(JobError::NotFound(job_id.to_string())));

    let app = build_router(test_state(mock));
    let resp: Response = app
        .oneshot(
            Request::get("/jobs/nonexistent-id")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn get_job_returns_status() {
    let mut mock = MockJobs::new();
    mock.expect_get_job().returning(|_| {
        let mut rec = make_job_record("job-3", JobType::Sync, JobStatus::Running);
        rec.progress = 0.5;
        rec.message = Some("halfway".into());
        Ok(rec)
    });

    let app = build_router(test_state(mock));
    let resp: Response = app
        .oneshot(Request::get("/jobs/job-3").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = response_json(resp).await;
    assert_eq!(v["progress"], 0.5);
}

#[tokio::test]
async fn cancel_job_returns_updated_status() {
    let mut mock = MockJobs::new();
    mock.expect_cancel_job().returning(|_| {
        let mut rec = make_job_record("job-4", JobType::Sync, JobStatus::CancelRequested);
        rec.cancel_requested = true;
        Ok(rec)
    });

    let app = build_router(test_state(mock));
    let resp: Response = app
        .oneshot(
            Request::post("/jobs/job-4/cancel")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = response_json(resp).await;
    assert!(v["cancel_requested"].as_bool().unwrap());
}

#[tokio::test]
async fn job_backend_unavailable_returns_503() {
    let mut mock = MockJobs::new();
    mock.expect_enqueue_sync_job()
        .returning(|_, _| Err(JobError::BackendUnavailable("redis down".into())));

    let app = build_router(test_state(mock));
    let body = json!({ "source": "/path/col.anki2" });
    let resp: Response = app
        .oneshot(
            Request::post("/jobs/sync")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

// ---- Removed surfaces ----

#[tokio::test]
async fn removed_sync_route_returns_404() {
    let app = build_router(test_state(MockJobs::new()));
    let resp: Response = app
        .oneshot(
            Request::post("/sync")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn removed_index_route_returns_404() {
    let app = build_router(test_state(MockJobs::new()));
    let resp: Response = app
        .oneshot(
            Request::post("/index")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn removed_search_route_returns_404() {
    let app = build_router(test_state(MockJobs::new()));
    let body = json!({ "query": "test query" });
    let resp: Response = app
        .oneshot(
            Request::post("/search")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn removed_topics_routes_return_404() {
    let app = build_router(test_state(MockJobs::new()));

    let topics = app
        .clone()
        .oneshot(Request::get("/topics").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(topics.status(), StatusCode::NOT_FOUND);

    let coverage = app
        .clone()
        .oneshot(
            Request::get("/topics/cs/algorithms/coverage")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(coverage.status(), StatusCode::NOT_FOUND);

    let gaps = app
        .oneshot(
            Request::get("/topics/cs/algorithms/gaps")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(gaps.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn removed_duplicates_route_returns_404() {
    let app = build_router(test_state(MockJobs::new()));
    let resp: Response = app
        .oneshot(Request::get("/duplicates").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn removed_index_info_route_returns_404() {
    let app = build_router(test_state(MockJobs::new()));
    let resp: Response = app
        .oneshot(Request::get("/index/info").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

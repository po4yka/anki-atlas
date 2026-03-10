use analytics::AnalyticsError;
use analytics::coverage::{GapType, TopicCoverage, TopicGap, WeakNote};
use analytics::duplicates::{DuplicateCluster, DuplicateDetail, DuplicateStats};
use analytics::labeling::LabelingStats;
use analytics::taxonomy::Taxonomy;
use anki_atlas_api::router::build_router;
use anki_atlas_api::schemas::SearchRequest;
use anki_atlas_api::services::{AnalyticsFacade, ApiServices, SearchFacade, build_app_state};
use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::Response;
use common::config::Settings;
use jobs::{
    IndexJobPayload, JobError, JobManager, JobPayload, JobRecord, JobStatus, JobType,
    SyncJobPayload,
};
use mockall::{Sequence, mock};
use search::error::SearchError;
use search::fts::LexicalMode;
use search::fusion::{FusionStats, SearchResult};
use search::service::{HybridSearchResult, SearchParams};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tower::ServiceExt;

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

mock! {
    pub Search {}

    #[async_trait]
    impl SearchFacade for Search {
        async fn search(
            &self,
            params: &SearchParams,
        ) -> Result<HybridSearchResult, SearchError>;
    }
}

mock! {
    pub Analytics {}

    #[async_trait]
    impl AnalyticsFacade for Analytics {
        async fn load_taxonomy(
            &self,
            yaml_path: Option<PathBuf>,
        ) -> Result<Taxonomy, AnalyticsError>;

        async fn label_notes(
            &self,
            yaml_path: Option<PathBuf>,
            min_confidence: f32,
        ) -> Result<LabelingStats, AnalyticsError>;

        async fn get_taxonomy_tree(
            &self,
            root_path: Option<String>,
        ) -> Result<Vec<Value>, AnalyticsError>;

        async fn get_coverage(
            &self,
            topic_path: String,
            include_subtree: bool,
        ) -> Result<Option<TopicCoverage>, AnalyticsError>;

        async fn get_gaps(
            &self,
            topic_path: String,
            min_coverage: i64,
        ) -> Result<Vec<TopicGap>, AnalyticsError>;

        async fn get_weak_notes(
            &self,
            topic_path: String,
            max_results: i64,
        ) -> Result<Vec<WeakNote>, AnalyticsError>;

        async fn find_duplicates(
            &self,
            threshold: f64,
            max_clusters: usize,
            deck_filter: Option<Vec<String>>,
            tag_filter: Option<Vec<String>>,
        ) -> Result<(Vec<DuplicateCluster>, DuplicateStats), AnalyticsError>;
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

fn lazy_pool() -> sqlx::PgPool {
    sqlx::postgres::PgPoolOptions::new()
        .connect_lazy("postgresql://localhost:5432/test")
        .expect("lazy pool")
}

fn test_app(
    mock_jobs: MockJobs,
    mock_search: MockSearch,
    mock_analytics: MockAnalytics,
) -> axum::Router {
    let services = ApiServices::new(
        lazy_pool(),
        Arc::new(mock_jobs),
        Arc::new(mock_search),
        Arc::new(mock_analytics),
    );
    build_router(build_app_state(test_settings().api(), services))
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

fn sample_search_result() -> HybridSearchResult {
    HybridSearchResult {
        results: vec![SearchResult {
            note_id: 1,
            rrf_score: 0.95,
            semantic_score: Some(0.9),
            semantic_rank: Some(1),
            fts_score: Some(0.8),
            fts_rank: Some(2),
            headline: Some("ownership".into()),
            rerank_score: Some(0.97),
            rerank_rank: Some(1),
        }],
        stats: FusionStats {
            semantic_only: 0,
            fts_only: 0,
            both: 1,
            total: 1,
        },
        query: "ownership".into(),
        filters_applied: HashMap::new(),
        lexical_mode: LexicalMode::Fts,
        lexical_fallback_used: false,
        query_suggestions: vec!["ownership and borrowing".into()],
        autocomplete_suggestions: vec!["ownership".into()],
        rerank_applied: true,
        rerank_model: Some("cross-encoder/test".into()),
        rerank_top_n: Some(10),
    }
}

async fn response_json(resp: Response) -> Value {
    let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}

#[tokio::test]
async fn build_router_accepts_mock_services() {
    let app = test_app(MockJobs::new(), MockSearch::new(), MockAnalytics::new());
    let resp = app
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn health_returns_200_with_version() {
    let app = test_app(MockJobs::new(), MockSearch::new(), MockAnalytics::new());
    let resp = app
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let v = response_json(resp).await;
    assert_eq!(v["status"], "healthy");
    assert!(v["version"].is_string());
}

#[tokio::test]
async fn ready_returns_200() {
    let app = test_app(MockJobs::new(), MockSearch::new(), MockAnalytics::new());
    let resp = app
        .oneshot(Request::get("/ready").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn enqueue_sync_job_returns_202() {
    let mut mock = MockJobs::new();
    mock.expect_enqueue_sync_job()
        .returning(|_, _| Ok(make_job_record("job-1", JobType::Sync, JobStatus::Queued)));

    let app = test_app(mock, MockSearch::new(), MockAnalytics::new());
    let body = json!({ "source": "/path/col.anki2" });
    let resp = app
        .oneshot(
            Request::post("/jobs/sync")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::ACCEPTED);
}

#[tokio::test]
async fn enqueue_index_job_returns_202() {
    let mut mock = MockJobs::new();
    mock.expect_enqueue_index_job()
        .returning(|_, _| Ok(make_job_record("job-2", JobType::Index, JobStatus::Queued)));

    let app = test_app(mock, MockSearch::new(), MockAnalytics::new());
    let resp = app
        .oneshot(
            Request::post("/jobs/index")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
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

    let app = test_app(mock, MockSearch::new(), MockAnalytics::new());
    let body = json!({
        "source": "/path/col.anki2",
        "run_at": "2026-03-11T10:00:00Z"
    });
    let resp = app
        .oneshot(
            Request::post("/jobs/sync")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
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

    let app = test_app(mock, MockSearch::new(), MockAnalytics::new());
    let resp = app
        .oneshot(Request::get("/jobs/missing").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn cancel_job_returns_updated_status() {
    let mut mock = MockJobs::new();
    mock.expect_cancel_job().returning(|_| {
        let mut rec = make_job_record("job-4", JobType::Sync, JobStatus::CancelRequested);
        rec.cancel_requested = true;
        Ok(rec)
    });

    let app = test_app(mock, MockSearch::new(), MockAnalytics::new());
    let resp = app
        .oneshot(
            Request::post("/jobs/job-4/cancel")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn search_forwards_filters_and_returns_typed_response() {
    let mut mock = MockSearch::new();
    mock.expect_search()
        .withf(|params| {
            params.query == "ownership"
                && params.limit == 10
                && params.semantic_only
                && !params.fts_only
                && params.rerank_override == Some(true)
                && params.rerank_top_n_override == Some(10)
                && params
                    .filters
                    .as_ref()
                    .and_then(|filters| filters.deck_names.as_ref())
                    == Some(&vec!["Rust".to_string()])
        })
        .returning(|_| Ok(sample_search_result()));

    let app = test_app(MockJobs::new(), mock, MockAnalytics::new());
    let body = SearchRequest {
        query: "ownership".into(),
        filters: Some(anki_atlas_api::schemas::SearchFiltersDto {
            deck_names: Some(vec!["Rust".into()]),
            ..Default::default()
        }),
        limit: 10,
        semantic_weight: 1.0,
        fts_weight: 0.5,
        semantic_only: true,
        fts_only: false,
        rerank_override: Some(true),
        rerank_top_n_override: Some(10),
    };

    let resp = app
        .oneshot(
            Request::post("/search")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = response_json(resp).await;
    assert_eq!(v["query"], "ownership");
    assert_eq!(v["results"][0]["sources"][0], "semantic");
    assert_eq!(v["lexical_mode"], "fts");
    assert_eq!(v["rerank_applied"], true);
}

#[tokio::test]
async fn search_rejects_invalid_flags_with_400() {
    let app = test_app(MockJobs::new(), MockSearch::new(), MockAnalytics::new());
    let body = json!({
        "query": "ownership",
        "semantic_only": true,
        "fts_only": true
    });
    let resp = app
        .oneshot(
            Request::post("/search")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn topics_returns_tree_for_root_path() {
    let mut analytics = MockAnalytics::new();
    analytics
        .expect_get_taxonomy_tree()
        .withf(|root| root.as_deref() == Some("cs"))
        .returning(|_| Ok(vec![json!({"path": "cs", "label": "CS"})]));

    let app = test_app(MockJobs::new(), MockSearch::new(), analytics);
    let resp = app
        .oneshot(
            Request::get("/topics?root_path=cs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = response_json(resp).await;
    assert_eq!(v["topics"][0]["path"], "cs");
}

#[tokio::test]
async fn topic_coverage_returns_404_for_missing_topic() {
    let mut analytics = MockAnalytics::new();
    analytics
        .expect_get_coverage()
        .withf(|path, include_subtree| path == "missing/topic" && *include_subtree)
        .returning(|_, _| Ok(None));

    let app = test_app(MockJobs::new(), MockSearch::new(), analytics);
    let resp = app
        .oneshot(
            Request::get("/topic-coverage?topic_path=missing/topic&include_subtree=true")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn topic_coverage_returns_metrics() {
    let mut analytics = MockAnalytics::new();
    analytics.expect_get_coverage().returning(|_, _| {
        Ok(Some(TopicCoverage {
            topic_id: 42,
            path: "cs/algorithms".into(),
            label: "Algorithms".into(),
            note_count: 4,
            subtree_count: 6,
            child_count: 2,
            covered_children: 1,
            mature_count: 2,
            avg_confidence: 0.8,
            weak_notes: 1,
            avg_lapses: 0.5,
        }))
    });

    let app = test_app(MockJobs::new(), MockSearch::new(), analytics);
    let resp = app
        .oneshot(
            Request::get("/topic-coverage?topic_path=cs/algorithms")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn topic_gaps_returns_typed_gap_type() {
    let mut analytics = MockAnalytics::new();
    let mut seq = Sequence::new();
    analytics
        .expect_get_coverage()
        .times(1)
        .in_sequence(&mut seq)
        .returning(|_, _| {
            Ok(Some(TopicCoverage {
                topic_id: 1,
                path: "cs".into(),
                label: "CS".into(),
                note_count: 0,
                subtree_count: 0,
                child_count: 0,
                covered_children: 0,
                mature_count: 0,
                avg_confidence: 0.0,
                weak_notes: 0,
                avg_lapses: 0.0,
            }))
        });
    analytics
        .expect_get_gaps()
        .times(1)
        .in_sequence(&mut seq)
        .withf(|path, min_coverage| path == "cs" && *min_coverage == 2)
        .returning(|_, _| {
            Ok(vec![TopicGap {
                topic_id: 10,
                path: "cs/networking".into(),
                label: "Networking".into(),
                description: None,
                gap_type: GapType::Missing,
                note_count: 0,
                threshold: 2,
                nearest_notes: vec![],
            }])
        });

    let app = test_app(MockJobs::new(), MockSearch::new(), analytics);
    let resp = app
        .oneshot(
            Request::get("/topic-gaps?topic_path=cs&min_coverage=2")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let v = response_json(resp).await;
    assert_eq!(v["gaps"][0]["gap_type"], "missing");
}

#[tokio::test]
async fn topic_weak_notes_uses_default_max_results() {
    let mut analytics = MockAnalytics::new();
    let mut seq = Sequence::new();
    analytics
        .expect_get_coverage()
        .times(1)
        .in_sequence(&mut seq)
        .returning(|_, _| {
            Ok(Some(TopicCoverage {
                topic_id: 1,
                path: "cs".into(),
                label: "CS".into(),
                note_count: 1,
                subtree_count: 1,
                child_count: 0,
                covered_children: 0,
                mature_count: 0,
                avg_confidence: 0.0,
                weak_notes: 0,
                avg_lapses: 0.0,
            }))
        });
    analytics
        .expect_get_weak_notes()
        .times(1)
        .in_sequence(&mut seq)
        .withf(|path, max_results| path == "cs" && *max_results == 20)
        .returning(|_, _| {
            Ok(vec![WeakNote {
                note_id: 5,
                topic_path: "cs".into(),
                confidence: 0.7,
                lapses: 3,
                fail_rate: Some(0.2),
                normalized_text: "preview".into(),
            }])
        });

    let app = test_app(MockJobs::new(), MockSearch::new(), analytics);
    let resp = app
        .oneshot(
            Request::get("/topic-weak-notes?topic_path=cs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn duplicates_forwards_query_filters() {
    let mut analytics = MockAnalytics::new();
    analytics
        .expect_find_duplicates()
        .withf(|threshold, max_clusters, deck_filter, tag_filter| {
            (*threshold - 0.9).abs() < f64::EPSILON
                && *max_clusters == 5
                && deck_filter.as_deref() == Some(&["Rust".to_string()][..])
                && tag_filter.as_deref() == Some(&["ownership".to_string()][..])
        })
        .returning(|_, _, _, _| {
            Ok((
                vec![DuplicateCluster {
                    representative_id: 1,
                    representative_text: "What is ownership?".into(),
                    duplicates: vec![DuplicateDetail {
                        note_id: 2,
                        similarity: 0.96,
                        text: "Explain ownership".into(),
                        deck_names: vec!["Rust".into()],
                        tags: vec!["ownership".into()],
                    }],
                    deck_names: vec!["Rust".into()],
                    tags: vec!["ownership".into()],
                }],
                DuplicateStats {
                    notes_scanned: 2,
                    clusters_found: 1,
                    total_duplicates: 1,
                    avg_cluster_size: 2.0,
                },
            ))
        });

    let app = test_app(MockJobs::new(), MockSearch::new(), analytics);
    let resp = app
        .oneshot(
            Request::get(
                "/duplicates?threshold=0.9&max_clusters=5&deck_filter[]=Rust&tag_filter[]=ownership",
            )
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn direct_sync_and_index_routes_remain_absent() {
    let app = test_app(MockJobs::new(), MockSearch::new(), MockAnalytics::new());

    let sync = app
        .clone()
        .oneshot(
            Request::post("/sync")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(sync.status(), StatusCode::NOT_FOUND);

    let index = app
        .oneshot(
            Request::post("/index")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(index.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn wildcard_topic_and_index_info_routes_remain_absent() {
    let app = test_app(MockJobs::new(), MockSearch::new(), MockAnalytics::new());

    let wildcard = app
        .clone()
        .oneshot(
            Request::get("/topics/cs/algorithms/coverage")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(wildcard.status(), StatusCode::NOT_FOUND);

    let index_info = app
        .oneshot(Request::get("/index/info").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(index_info.status(), StatusCode::NOT_FOUND);
}

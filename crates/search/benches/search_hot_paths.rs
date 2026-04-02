use std::collections::HashMap;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use perf_support::{DatasetProfile, SeedManifest, seed_postgres_only};
use search::error::SearchError;
use search::fts::SearchFilters;
use search::reranker::Reranker;
use search::service::{SearchMode, SearchParams, SearchService};
use sqlx::postgres::PgPoolOptions;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

struct BenchVectorRepo {
    results: Vec<indexer::qdrant::ScoredNote>,
}

impl BenchVectorRepo {
    fn new(results: Vec<(i64, f32)>) -> Self {
        Self {
            results: results
                .into_iter()
                .map(|(note_id, score)| indexer::qdrant::ScoredNote { note_id, score })
                .collect(),
        }
    }
}

#[async_trait::async_trait]
impl indexer::qdrant::VectorRepository for BenchVectorRepo {
    async fn ensure_collection(
        &self,
        _dimension: usize,
    ) -> Result<bool, indexer::qdrant::VectorStoreError> {
        Ok(false)
    }

    async fn upsert_vectors(
        &self,
        _vectors: &[Vec<f32>],
        _payloads: &[indexer::qdrant::NotePayload],
        _sparse_vectors: Option<&[indexer::qdrant::SparseVector]>,
    ) -> Result<usize, indexer::qdrant::VectorStoreError> {
        Ok(0)
    }

    async fn delete_vectors(
        &self,
        _note_ids: &[i64],
    ) -> Result<usize, indexer::qdrant::VectorStoreError> {
        Ok(0)
    }

    async fn get_existing_hashes(
        &self,
        _note_ids: &[i64],
    ) -> Result<HashMap<i64, String>, indexer::qdrant::VectorStoreError> {
        Ok(HashMap::new())
    }

    async fn search(
        &self,
        _query_vector: &[f32],
        _query_sparse: Option<&indexer::qdrant::SparseVector>,
        _limit: usize,
        _filters: &indexer::qdrant::SearchFilters,
    ) -> Result<Vec<indexer::qdrant::ScoredNote>, indexer::qdrant::VectorStoreError> {
        Ok(self.results.clone())
    }

    async fn find_similar_to_note(
        &self,
        _note_id: i64,
        _limit: usize,
        _min_score: f32,
        _deck_names: Option<&[String]>,
        _tags: Option<&[String]>,
    ) -> Result<Vec<indexer::qdrant::ScoredNote>, indexer::qdrant::VectorStoreError> {
        Ok(Vec::new())
    }

    async fn close(&self) -> Result<(), indexer::qdrant::VectorStoreError> {
        Ok(())
    }
}

struct BenchReranker;

#[async_trait::async_trait]
impl Reranker for BenchReranker {
    fn model_name(&self) -> &str {
        "bench/reranker"
    }

    async fn rerank(
        &self,
        _query: &str,
        documents: &[(i64, String)],
    ) -> Result<Vec<(i64, f64)>, SearchError> {
        Ok(documents
            .iter()
            .enumerate()
            .map(|(index, (note_id, _))| (*note_id, 1.0 - index as f64 * 0.01))
            .collect())
    }
}

struct BenchFixture {
    pool: sqlx::PgPool,
    manifest: SeedManifest,
    _container: ContainerAsync<Postgres>,
}

async fn setup_fixture() -> BenchFixture {
    let container = Postgres::default()
        .start()
        .await
        .expect("postgres container");
    let host = container.get_host().await.expect("postgres host");
    let port = container
        .get_host_port_ipv4(5432)
        .await
        .expect("postgres port");
    let url = format!("postgresql://postgres:postgres@{host}:{port}/postgres");
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&url)
        .await
        .expect("connect Postgres");
    let manifest = seed_postgres_only(&pool, DatasetProfile::Pr)
        .await
        .expect("seed search bench fixture");

    BenchFixture {
        pool,
        manifest,
        _container: container,
    }
}

fn build_service(
    pool: sqlx::PgPool,
    vector_results: Vec<indexer::qdrant::ScoredNote>,
    reranker: Option<BenchReranker>,
) -> SearchService<
    indexer::embeddings::DeterministicEmbeddingProvider,
    Arc<BenchVectorRepo>,
    BenchReranker,
> {
    SearchService::new(
        indexer::embeddings::DeterministicEmbeddingProvider::new(384),
        Arc::new(BenchVectorRepo::new(vector_results)),
        reranker,
        pool,
        true,
        10,
    )
}

fn bench_search_hot_paths(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("bench runtime");
    let fixture = runtime.block_on(setup_fixture());
    let query = fixture.manifest.search_queries[0].clone();

    let fts_only_service = build_service(fixture.pool.clone(), Vec::new(), None);
    let hybrid_service = build_service(
        fixture.pool.clone(),
        vec![
            indexer::qdrant::ScoredNote {
                note_id: 1,
                score: 0.99,
            },
            indexer::qdrant::ScoredNote {
                note_id: 2,
                score: 0.95,
            },
            indexer::qdrant::ScoredNote {
                note_id: 3,
                score: 0.91,
            },
            indexer::qdrant::ScoredNote {
                note_id: 4,
                score: 0.88,
            },
            indexer::qdrant::ScoredNote {
                note_id: 5,
                score: 0.84,
            },
        ],
        None,
    );
    let rerank_service = build_service(
        fixture.pool.clone(),
        vec![
            indexer::qdrant::ScoredNote {
                note_id: 1,
                score: 0.99,
            },
            indexer::qdrant::ScoredNote {
                note_id: 2,
                score: 0.95,
            },
            indexer::qdrant::ScoredNote {
                note_id: 3,
                score: 0.91,
            },
            indexer::qdrant::ScoredNote {
                note_id: 4,
                score: 0.88,
            },
            indexer::qdrant::ScoredNote {
                note_id: 5,
                score: 0.84,
            },
        ],
        Some(BenchReranker),
    );

    let mut group = c.benchmark_group("search_hot_paths");

    group.bench_function("fts_only", |b| {
        let params = SearchParams {
            query: query.clone(),
            search_mode: SearchMode::FtsOnly,
            ..Default::default()
        };
        b.to_async(&runtime).iter(|| async {
            fts_only_service.search(&params).await.expect("fts search");
        });
    });

    group.bench_function("hybrid", |b| {
        let params = SearchParams {
            query: query.clone(),
            ..Default::default()
        };
        b.to_async(&runtime).iter(|| async {
            hybrid_service.search(&params).await.expect("hybrid search");
        });
    });

    group.bench_function("semantic_only", |b| {
        let params = SearchParams {
            query: query.clone(),
            search_mode: SearchMode::SemanticOnly,
            ..Default::default()
        };
        b.to_async(&runtime).iter(|| async {
            hybrid_service
                .search(&params)
                .await
                .expect("semantic-only search");
        });
    });

    group.bench_function("filtered", |b| {
        let params = SearchParams {
            query: fixture.manifest.search_queries[1].clone(),
            filters: Some(SearchFilters {
                deck_names: Some(vec![fixture.manifest.duplicate_deck.clone()]),
                tags: Some(vec![fixture.manifest.search_queries[1].clone()]),
                min_reps: Some(10),
                ..Default::default()
            }),
            ..Default::default()
        };
        b.to_async(&runtime).iter(|| async {
            hybrid_service
                .search(&params)
                .await
                .expect("filtered search");
        });
    });

    group.bench_function("rerank_document_prep", |b| {
        let params = SearchParams {
            query: query.clone(),
            rerank_override: Some(true),
            rerank_top_n_override: Some(5),
            ..Default::default()
        };
        b.to_async(&runtime).iter(|| async {
            rerank_service.search(&params).await.expect("rerank search");
        });
    });

    group.finish();
}

criterion_group!(benches, bench_search_hot_paths);
criterion_main!(benches);

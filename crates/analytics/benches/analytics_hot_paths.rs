use std::collections::HashMap;
use std::sync::Arc;

use analytics::coverage::{get_coverage_tree, get_topic_coverage, get_topic_gaps, get_weak_notes};
use analytics::duplicates::DuplicateDetector;
use criterion::{Criterion, criterion_group, criterion_main};
use perf_support::{DatasetProfile, SeedManifest, seed_postgres_only};
use sqlx::postgres::PgPoolOptions;
use testcontainers::ContainerAsync;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

struct DuplicateVectorRepo;

#[async_trait::async_trait]
impl indexer::qdrant::VectorRepository for DuplicateVectorRepo {
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
        Ok(Vec::new())
    }

    async fn find_similar_to_note(
        &self,
        note_id: i64,
        _limit: usize,
        _min_score: f32,
        _deck_names: Option<&[String]>,
        _tags: Option<&[String]>,
    ) -> Result<Vec<indexer::qdrant::ScoredNote>, indexer::qdrant::VectorStoreError> {
        if !(1..=75).contains(&note_id) {
            return Ok(Vec::new());
        }

        let cluster_index = (note_id - 1) / 3;
        let cluster_start = cluster_index * 3 + 1;
        Ok((cluster_start..cluster_start + 3)
            .filter(|other| *other != note_id)
            .map(|other| indexer::qdrant::ScoredNote {
                note_id: other,
                score: 0.97,
            })
            .collect())
    }

    async fn close(&self) -> Result<(), indexer::qdrant::VectorStoreError> {
        Ok(())
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
        .expect("seed analytics bench fixture");

    BenchFixture {
        pool,
        manifest,
        _container: container,
    }
}

fn bench_analytics_hot_paths(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("bench runtime");
    let fixture = runtime.block_on(setup_fixture());
    let root = fixture.manifest.root_topics[0].clone();
    let branch = fixture.manifest.branch_topics[0].clone();
    let detector = DuplicateDetector::new(Arc::new(DuplicateVectorRepo), fixture.pool.clone());

    let mut group = c.benchmark_group("analytics_hot_paths");

    group.bench_function("topic_coverage", |b| {
        b.to_async(&runtime).iter(|| async {
            get_topic_coverage(&fixture.pool, &branch, true)
                .await
                .expect("topic coverage");
        });
    });

    group.bench_function("topic_gaps", |b| {
        b.to_async(&runtime).iter(|| async {
            get_topic_gaps(&fixture.pool, &root, 4)
                .await
                .expect("topic gaps");
        });
    });

    group.bench_function("weak_notes", |b| {
        b.to_async(&runtime).iter(|| async {
            get_weak_notes(&fixture.pool, &root, 25, 0.1)
                .await
                .expect("weak notes");
        });
    });

    group.bench_function("taxonomy_tree", |b| {
        b.to_async(&runtime).iter(|| async {
            get_coverage_tree(&fixture.pool, Some(&root))
                .await
                .expect("taxonomy tree");
        });
    });

    group.bench_function("find_duplicates", |b| {
        b.to_async(&runtime).iter(|| async {
            detector
                .find_duplicates(0.95, 25, None, None)
                .await
                .expect("duplicates");
        });
    });

    group.finish();
}

criterion_group!(benches, bench_analytics_hot_paths);
criterion_main!(benches);

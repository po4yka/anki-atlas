# Performance Testing

Anki Atlas now has two performance layers:

- Criterion microbenchmarks for crate hot paths
- a Goose-based `perf-harness` for end-to-end API and worker load tests

All performance runs force mock embeddings so they stay deterministic and do not require external model credentials.

## Prerequisites

- Docker
- PostgreSQL, Qdrant, and Redis reachable through the standard `ANKIATLAS_*` environment variables

The local stack from `infra/docker-compose.yml` is sufficient:

```bash
docker compose -f infra/docker-compose.yml up -d
```

## Seed a profile

Seed the deterministic dataset for the target profile before starting the API and worker:

```bash
cargo run -p perf-harness -- --profile pr --prepare-only
```

Profiles:

- `pr`: 1,000 notes, 120 topics, 25 duplicate clusters
- `nightly`: 10,000 notes, 1,000 topics, 250 duplicate clusters

The seed manifest is written to `target/perf/seed-manifest-<profile>.json`.

## Run the API smoke load

Start the API and worker in separate terminals:

```bash
cargo run --bin anki-atlas-api
ANKIATLAS_ENABLE_EXPERIMENTAL_JOB_WORKER=1 cargo run --bin anki-atlas-worker
```

Then run the harness:

```bash
cargo run -p perf-harness -- --profile pr --scenario full
```

Optional flags:

- `--base-url http://127.0.0.1:8000`
- `--report-json target/perf/custom-report.json`

The default report path is `target/perf/<profile>-<scenario>.json`.

## Benchmark hot paths

Search benchmarks:

```bash
cargo bench -p search --bench search_hot_paths
```

Analytics benchmarks:

```bash
cargo bench -p analytics --bench analytics_hot_paths
```

These benches use seeded Postgres fixtures plus stubbed vector/rerank dependencies. They intentionally do not hit real Qdrant or Redis.

## Interpreting reports

`perf-harness` reports:

- aggregated read-path p95 latency
- aggregated job-path p95 latency
- request error rates for read and job groups
- worker terminalization ratio inside the configured SLA window

PR smoke runs fail when any of these thresholds are exceeded:

- read error rate `> 1%`
- read p95 `> 1.5s`
- job error rate `> 1%`
- job p95 `> 750ms`
- fewer than 95% of tracked jobs reaching terminal state within 5 seconds

Nightly runs are report-only and upload artifacts without enforcing historical regression gates.

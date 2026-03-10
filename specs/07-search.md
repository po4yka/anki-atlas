# Spec: crate `search`

## Source Reference

Current Rust implementation: `crates/search/`
Historical rewrite input: `packages/search/` (fts.py, fusion.py, reranker.py, service.py)

## Purpose

Provides hybrid search over Anki notes by combining PostgreSQL full-text search (with fuzzy and autocomplete fallbacks) with semantic vector search, fusing results via Reciprocal Rank Fusion (RRF), and optionally reranking the top candidates with a cross-encoder model. The `SearchService` orchestrates the full pipeline.

## Dependencies

```toml
[dependencies]
common = { path = "../common" }
database = { path = "../database" }
indexer = { path = "../indexer" }
async-trait = "0.1"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
sqlx = { version = "0.8", features = ["runtime-tokio", "postgres"] }
thiserror = "2"
tracing = "0.1"
regex = "1"

[dev-dependencies]
mockall = "0.13"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Public API

### Full-Text Search (`src/fts.rs`)

```rust
use serde::{Deserialize, Serialize};

/// Source of a lexical search result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FtsSource {
    Fts,
    Fuzzy,
    Autocomplete,
}

/// Single FTS hit.
#[derive(Debug, Clone, Serialize)]
pub struct FtsResult {
    pub note_id: i64,
    pub rank: f64,
    pub headline: Option<String>,
    pub source: FtsSource,
}

/// Lexical search mode that was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalMode {
    Fts,
    Fuzzy,
    Autocomplete,
    None,
}

/// Bundle of lexical results with fallback metadata.
#[derive(Debug, Clone, Serialize)]
pub struct LexicalSearchResult {
    pub results: Vec<FtsResult>,
    pub mode: LexicalMode,
    pub used_fallback: bool,
    pub query_suggestions: Vec<String>,
    pub autocomplete_suggestions: Vec<String>,
}

/// Filters for search queries (shared between FTS and hybrid).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilters {
    pub deck_names: Option<Vec<String>>,
    pub deck_names_exclude: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub tags_exclude: Option<Vec<String>>,
    pub model_ids: Option<Vec<i64>>,
    pub min_ivl: Option<i32>,
    pub max_lapses: Option<i32>,
    pub min_reps: Option<i32>,
}

/// Execute lexical search with FTS -> fuzzy -> autocomplete fallback chain.
pub async fn search_lexical(
    pool: &sqlx::PgPool,
    query: &str,
    filters: Option<&SearchFilters>,
    limit: i64,
) -> Result<LexicalSearchResult, SearchError>;
```

### Reciprocal Rank Fusion (`src/fusion.rs`)

```rust
use serde::Serialize;

/// Fused search result with score breakdown from all retrieval stages.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub note_id: i64,
    pub rrf_score: f64,
    pub semantic_score: Option<f64>,
    pub semantic_rank: Option<usize>,
    pub fts_score: Option<f64>,
    pub fts_rank: Option<usize>,
    pub headline: Option<String>,
    pub rerank_score: Option<f64>,
    pub rerank_rank: Option<usize>,
}

impl SearchResult {
    /// Return list of contributing sources ("semantic", "fts").
    pub fn sources(&self) -> Vec<&'static str>;
}

/// Statistics about the fusion operation.
#[derive(Debug, Clone, Default, Serialize)]
pub struct FusionStats {
    pub semantic_only: usize,
    pub fts_only: usize,
    pub both: usize,
    pub total: usize,
}

/// Fuse semantic and FTS results using Reciprocal Rank Fusion.
///
/// RRF score = sum( weight / (k + rank) ) across sources.
pub fn reciprocal_rank_fusion(
    semantic_results: &[(i64, f64)],
    fts_results: &[(i64, f64, Option<String>)],
    k: usize,
    limit: usize,
    semantic_weight: f64,
    fts_weight: f64,
) -> (Vec<SearchResult>, FusionStats);
```

### Reranker (`src/reranker.rs`)

```rust
/// Trait for second-stage reranking implementations.
#[async_trait::async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait Reranker: Send + Sync {
    /// Score (document_id, text) pairs against a query.
    /// Returns (document_id, score) pairs in arbitrary order.
    async fn rerank(
        &self,
        query: &str,
        documents: &[(i64, String)],
    ) -> Result<Vec<(i64, f64)>, SearchError>;
}

/// Cross-encoder reranker that calls an external inference endpoint.
pub struct CrossEncoderReranker {
    model_name: String,
    batch_size: usize,
    client: reqwest::Client,
    endpoint: String,
}

impl CrossEncoderReranker {
    pub fn new(
        model_name: impl Into<String>,
        batch_size: usize,
        endpoint: impl Into<String>,
    ) -> Self;
}
```

### Search Service (`src/service.rs`)

```rust
use std::collections::HashMap;
use serde::Serialize;

/// Complete hybrid search result.
#[derive(Debug, Clone, Serialize)]
pub struct HybridSearchResult {
    pub results: Vec<SearchResult>,
    pub stats: FusionStats,
    pub query: String,
    pub filters_applied: HashMap<String, serde_json::Value>,
    pub lexical_mode: String,
    pub lexical_fallback_used: bool,
    pub query_suggestions: Vec<String>,
    pub autocomplete_suggestions: Vec<String>,
    pub rerank_applied: bool,
    pub rerank_model: Option<String>,
    pub rerank_top_n: Option<usize>,
}

/// Detailed note information for result enrichment.
#[derive(Debug, Clone, Serialize)]
pub struct NoteDetail {
    pub note_id: i64,
    pub model_id: i64,
    pub normalized_text: String,
    pub tags: Vec<String>,
    pub deck_names: Vec<String>,
    pub mature: bool,
    pub lapses: i32,
    pub reps: i32,
}

/// Search errors.
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("embedding error: {0}")]
    Embedding(#[from] indexer::embeddings::EmbeddingError),
    #[error("vector store error: {0}")]
    VectorStore(#[from] indexer::qdrant::VectorStoreError),
    #[error("rerank failed: {0}")]
    Rerank(String),
}

/// Search service with trait-based DI.
pub struct SearchService<E, V, R>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
    R: Reranker,
{
    embedding: E,
    vector_repo: V,
    reranker: Option<R>,
    db: sqlx::PgPool,
    rerank_enabled: bool,
    rerank_top_n: usize,
}

impl<E, V, R> SearchService<E, V, R>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
    R: Reranker,
{
    pub fn new(
        embedding: E,
        vector_repo: V,
        reranker: Option<R>,
        db: sqlx::PgPool,
        rerank_enabled: bool,
        rerank_top_n: usize,
    ) -> Self;

    /// Execute hybrid search: semantic + FTS -> RRF fusion -> optional rerank.
    pub async fn search(
        &self,
        query: &str,
        filters: Option<&SearchFilters>,
        limit: usize,
        semantic_weight: f64,
        fts_weight: f64,
        semantic_only: bool,
        fts_only: bool,
        rerank_override: Option<bool>,
        rerank_top_n_override: Option<usize>,
    ) -> Result<HybridSearchResult, SearchError>;

    /// Fetch note details for a list of IDs (for reranking / enrichment).
    pub async fn get_notes_details(
        &self,
        note_ids: &[i64],
    ) -> Result<HashMap<i64, NoteDetail>, SearchError>;
}
```

## Internal Details

### FTS fallback chain
1. **FTS**: `websearch_to_tsquery('english', query)` against `to_tsvector('english', normalized_text)`. Uses `ts_rank` for scoring and `ts_headline` for snippet generation.
2. **Fuzzy**: trigram similarity via `similarity()` and `word_similarity()` with thresholds 0.15 and 0.2 respectively. Uses the `%%` operator.
3. **Autocomplete**: prefix matching on individual tokens via `lower(token) LIKE '{prefix}%'`.
4. At each stage, if results are non-empty, return immediately. Query suggestions and autocomplete suggestions are computed alongside fuzzy stage.

### SQL filter application
- Filters on `tags`, `tags_exclude`, `model_ids` apply to the `notes` table directly.
- Filters on `deck_names`, `deck_names_exclude`, `min_ivl`, `max_lapses`, `min_reps` require LEFT JOINing `cards` and `decks` tables. A `GROUP BY` is added when the card join is present.
- In Rust, build the query dynamically using `sqlx::QueryBuilder` or string assembly with bound parameters. Avoid SQL injection by always using parameterized queries.

### RRF algorithm
- For each note_id appearing in either result set:
  - `rrf_score += semantic_weight / (k + semantic_rank)` if present in semantic results
  - `rrf_score += fts_weight / (k + fts_rank)` if present in FTS results
- Ranks are 1-indexed from the sorted input lists.
- Sort by `rrf_score` descending, take top `limit`.

### Reranking
- If enabled, over-fetch candidates: `candidate_limit = max(limit, rerank_top_n) * 2`.
- Fetch `NoteDetail.normalized_text` for top-N candidates from PostgreSQL.
- Pass (note_id, text) pairs to the `Reranker` trait.
- Re-sort by rerank score. Unscored candidates go after scored ones, then remaining tail from RRF.
- Log a warning (once) if reranker is unavailable and fall back to RRF ordering.

### Candidate limit calculation
- Without rerank: `candidate_limit = limit`
- With rerank: `candidate_limit = max(limit, rerank_top_n)`
- Retrieval limit (per source): `candidate_limit * 2`

## Acceptance Criteria

- [ ] `cargo test -p search` passes
- [ ] `cargo clippy -p search -- -D warnings` clean
- [ ] All public types are `Send + Sync`
- [ ] `reciprocal_rank_fusion` with empty inputs returns empty results and zeroed stats
- [ ] `reciprocal_rank_fusion` with overlapping results sets `both` count correctly
- [ ] RRF score formula matches `sum(weight / (k + rank))` -- verified with known inputs
- [ ] `SearchResult::sources()` returns `["semantic"]`, `["fts"]`, or `["semantic", "fts"]`
- [ ] `search_lexical` returns `LexicalMode::None` for blank queries
- [ ] FTS fallback chain: FTS -> fuzzy -> autocomplete tested with mock DB
- [ ] `SearchService` is generic over `EmbeddingProvider + VectorRepository + Reranker`
- [ ] `semantic_only = true` skips FTS; `fts_only = true` skips semantic
- [ ] Reranker failure degrades gracefully (returns RRF results with `rerank_applied = false`)
- [ ] SQL filter builder produces correct parameterized queries (no string interpolation of user input)

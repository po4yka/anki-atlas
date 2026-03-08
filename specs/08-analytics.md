# Spec: crate `analytics`

## Source Reference

Python: `packages/analytics/` (taxonomy.py, coverage.py, labeling.py, duplicates.py, service.py)

## Purpose

Provides topic taxonomy management (YAML/DB), coverage analysis with gap detection, embedding-based note-topic labeling, and near-duplicate detection using vector similarity clustering. The `AnalyticsService` facade ties these together for CLI and API consumers.

## Dependencies

```toml
[dependencies]
common = { path = "../common" }
database = { path = "../database" }
indexer = { path = "../indexer" }
async-trait = "0.1"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
serde_yaml = "0.9"
sqlx = { version = "0.8", features = ["runtime-tokio", "postgres"] }
thiserror = "2"
tracing = "0.1"

[dev-dependencies]
mockall = "0.13"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
tempfile = "3"
```

## Public API

### Taxonomy (`src/taxonomy.rs`)

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A topic node in the taxonomy tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    /// Slash-delimited path (e.g. "programming/python/async").
    pub path: String,
    /// Human-readable label.
    pub label: String,
    /// Optional description.
    pub description: Option<String>,
    /// Database ID (populated after DB insert).
    pub topic_id: Option<i64>,
    /// Child topics (populated during tree construction).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<Topic>,
}

impl Topic {
    /// Parent path (everything before last '/'), or None for root topics.
    pub fn parent_path(&self) -> Option<&str>;

    /// Depth = number of '/' in path.
    pub fn depth(&self) -> usize;

    /// Short name = last segment of path.
    pub fn name(&self) -> &str;
}

/// Complete taxonomy with lookup index.
#[derive(Debug, Clone, Default)]
pub struct Taxonomy {
    /// path -> Topic lookup.
    pub topics: HashMap<String, Topic>,
    /// Root topics (depth 0).
    pub roots: Vec<Topic>,
}

impl Taxonomy {
    /// Get topic by path.
    pub fn get(&self, path: &str) -> Option<&Topic>;

    /// All topics in depth-first order.
    pub fn all_topics(&self) -> Vec<&Topic>;

    /// All topics under a given path (inclusive).
    pub fn subtree(&self, path: &str) -> Vec<&Topic>;
}

/// Load taxonomy from a YAML file.
///
/// Expected format:
/// ```yaml
/// topics:
///   - path: programming
///     label: Programming
///     children:
///       - path: programming/python
///         label: Python
/// ```
pub fn load_taxonomy_from_yaml(path: &std::path::Path) -> Result<Taxonomy, AnalyticsError>;

/// Sync taxonomy to database (upsert by path). Returns path -> topic_id map.
pub async fn sync_taxonomy_to_db(
    pool: &sqlx::PgPool,
    taxonomy: &Taxonomy,
) -> Result<HashMap<String, i64>, AnalyticsError>;

/// Load taxonomy from database, reconstructing tree structure.
pub async fn load_taxonomy_from_db(pool: &sqlx::PgPool) -> Result<Taxonomy, AnalyticsError>;

/// Get a single topic by path from database.
pub async fn get_topic_by_path(pool: &sqlx::PgPool, path: &str) -> Result<Option<Topic>, AnalyticsError>;
```

### Coverage (`src/coverage.rs`)

```rust
use serde::Serialize;

/// Coverage metrics for a topic.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TopicCoverage {
    pub topic_id: i64,
    pub path: String,
    pub label: String,
    pub note_count: i64,
    pub subtree_count: i64,
    pub child_count: i64,
    pub covered_children: i64,
    pub mature_count: i64,
    pub avg_confidence: f64,
    pub weak_notes: i64,
    pub avg_lapses: f64,
}

/// A gap in topic coverage.
#[derive(Debug, Clone, Serialize)]
pub struct TopicGap {
    pub topic_id: i64,
    pub path: String,
    pub label: String,
    pub description: Option<String>,
    /// "missing" (0 notes) or "undercovered" (< threshold).
    pub gap_type: GapType,
    pub note_count: i64,
    pub threshold: i64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nearest_notes: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GapType {
    Missing,
    Undercovered,
}

/// A note with weakness signals.
#[derive(Debug, Clone, Serialize)]
pub struct WeakNote {
    pub note_id: i64,
    pub topic_path: String,
    pub confidence: f64,
    pub lapses: i32,
    pub fail_rate: Option<f64>,
    /// Truncated to 200 chars.
    pub normalized_text: String,
}

/// Get coverage metrics for a topic (optionally including subtree).
pub async fn get_topic_coverage(
    pool: &sqlx::PgPool,
    topic_path: &str,
    include_subtree: bool,
) -> Result<Option<TopicCoverage>, AnalyticsError>;

/// Find gaps in topic coverage under a root path.
pub async fn get_topic_gaps(
    pool: &sqlx::PgPool,
    topic_path: &str,
    min_coverage: i64,
) -> Result<Vec<TopicGap>, AnalyticsError>;

/// Get weak notes (high lapse rate) in a topic subtree.
pub async fn get_weak_notes(
    pool: &sqlx::PgPool,
    topic_path: &str,
    max_results: i64,
    min_fail_rate: f64,
) -> Result<Vec<WeakNote>, AnalyticsError>;

/// Get coverage tree for all topics (optionally filtered by root path).
pub async fn get_coverage_tree(
    pool: &sqlx::PgPool,
    root_path: Option<&str>,
) -> Result<Vec<serde_json::Value>, AnalyticsError>;
```

### Labeling (`src/labeling.rs`)

```rust
use serde::Serialize;

/// A topic assignment for a note.
#[derive(Debug, Clone, Serialize)]
pub struct TopicAssignment {
    pub note_id: i64,
    pub topic_id: i64,
    pub topic_path: String,
    pub confidence: f64,
    /// Labeling method (e.g. "embedding").
    pub method: String,
}

/// Statistics from a labeling operation.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LabelingStats {
    pub notes_processed: usize,
    pub assignments_created: usize,
    pub topics_matched: usize,
}

/// Cosine similarity between two equal-length f32 vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32;

/// Topic labeler. Generic over embedding provider.
pub struct TopicLabeler<E: indexer::embeddings::EmbeddingProvider> {
    embedding: E,
    db: sqlx::PgPool,
}

impl<E: indexer::embeddings::EmbeddingProvider> TopicLabeler<E> {
    pub fn new(embedding: E, db: sqlx::PgPool) -> Self;

    /// Embed all topic descriptions/labels. Returns path -> embedding vector.
    pub async fn embed_topics(
        &self,
        taxonomy: &super::taxonomy::Taxonomy,
    ) -> Result<std::collections::HashMap<String, Vec<f32>>, AnalyticsError>;

    /// Label all notes in database with matching topics.
    pub async fn label_notes(
        &self,
        taxonomy: &super::taxonomy::Taxonomy,
        min_confidence: f32,
        max_topics_per_note: usize,
        batch_size: usize,
    ) -> Result<LabelingStats, AnalyticsError>;

    /// Label a single note.
    pub async fn label_single_note(
        &self,
        note_id: i64,
        taxonomy: &super::taxonomy::Taxonomy,
        topic_embeddings: Option<&std::collections::HashMap<String, Vec<f32>>>,
        min_confidence: f32,
        max_topics: usize,
    ) -> Result<Vec<TopicAssignment>, AnalyticsError>;
}
```

### Duplicate Detection (`src/duplicates.rs`)

```rust
use serde::Serialize;

/// A single duplicate note within a cluster.
#[derive(Debug, Clone, Serialize)]
pub struct DuplicateDetail {
    pub note_id: i64,
    pub similarity: f64,
    pub text: String,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
}

/// A cluster of near-duplicate notes.
#[derive(Debug, Clone, Serialize)]
pub struct DuplicateCluster {
    pub representative_id: i64,
    pub representative_text: String,
    pub duplicates: Vec<DuplicateDetail>,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
}

impl DuplicateCluster {
    /// Total notes: representative + duplicates.
    pub fn size(&self) -> usize;
}

/// Statistics from duplicate detection.
#[derive(Debug, Clone, Default, Serialize)]
pub struct DuplicateStats {
    pub notes_scanned: usize,
    pub clusters_found: usize,
    pub total_duplicates: usize,
    pub avg_cluster_size: f64,
}

/// Duplicate detector. Uses VectorRepository for similarity search.
pub struct DuplicateDetector<V: indexer::qdrant::VectorRepository> {
    vector_repo: V,
    db: sqlx::PgPool,
}

impl<V: indexer::qdrant::VectorRepository> DuplicateDetector<V> {
    pub fn new(vector_repo: V, db: sqlx::PgPool) -> Self;

    /// Find clusters of near-duplicate notes.
    pub async fn find_duplicates(
        &self,
        threshold: f64,
        max_clusters: usize,
        deck_filter: Option<&[String]>,
        tag_filter: Option<&[String]>,
    ) -> Result<(Vec<DuplicateCluster>, DuplicateStats), AnalyticsError>;
}
```

### Analytics Service (`src/service.rs`)

```rust
/// Facade aggregating taxonomy, coverage, labeling, and duplicate detection.
pub struct AnalyticsService<E, V>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
{
    embedding: E,
    vector_repo: V,
    db: sqlx::PgPool,
}

impl<E, V> AnalyticsService<E, V>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
{
    pub fn new(embedding: E, vector_repo: V, db: sqlx::PgPool) -> Self;

    /// Load taxonomy from YAML (syncing to DB) or from DB.
    pub async fn load_taxonomy(
        &self,
        yaml_path: Option<&std::path::Path>,
    ) -> Result<Taxonomy, AnalyticsError>;

    /// Label all notes with topics.
    pub async fn label_notes(
        &self,
        taxonomy: Option<&Taxonomy>,
        min_confidence: f32,
    ) -> Result<LabelingStats, AnalyticsError>;

    pub async fn get_coverage(
        &self,
        topic_path: &str,
        include_subtree: bool,
    ) -> Result<Option<TopicCoverage>, AnalyticsError>;

    pub async fn get_gaps(
        &self,
        topic_path: &str,
        min_coverage: i64,
    ) -> Result<Vec<TopicGap>, AnalyticsError>;

    pub async fn get_weak_notes(
        &self,
        topic_path: &str,
        max_results: i64,
    ) -> Result<Vec<WeakNote>, AnalyticsError>;

    pub async fn find_duplicates(
        &self,
        threshold: f64,
        max_clusters: usize,
        deck_filter: Option<&[String]>,
        tag_filter: Option<&[String]>,
    ) -> Result<(Vec<DuplicateCluster>, DuplicateStats), AnalyticsError>;

    pub async fn get_taxonomy_tree(
        &self,
        root_path: Option<&str>,
    ) -> Result<Vec<serde_json::Value>, AnalyticsError>;
}

/// Analytics error enum.
#[derive(Debug, thiserror::Error)]
pub enum AnalyticsError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("embedding error: {0}")]
    Embedding(#[from] indexer::embeddings::EmbeddingError),
    #[error("vector store error: {0}")]
    VectorStore(#[from] indexer::qdrant::VectorStoreError),
    #[error("yaml parse error: {0}")]
    YamlParse(#[from] serde_yaml::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("topic not found: {0}")]
    TopicNotFound(String),
}
```

## Internal Details

### Taxonomy tree reconstruction from DB
1. Query all topics sorted by `path`.
2. Build flat `HashMap<String, Topic>`.
3. Second pass: for each topic, find parent by `parent_path()`. If parent exists in map, append as child. If no parent, add to `roots`.

### YAML taxonomy format
```yaml
topics:
  - path: programming
    label: Programming
    description: General programming concepts
    children:
      - path: programming/python
        label: Python
```
Recursively parse, registering each topic in the flat map and wiring parent-child links.

### Coverage SQL queries
- Direct note count: `COUNT(DISTINCT nt.note_id)` from `note_topics nt JOIN topics t` where `t.path = ?` or `t.path LIKE '{path}/%'`.
- Maturity: `COUNT(DISTINCT CASE WHEN c.ivl >= 21 THEN nt.note_id END)`.
- Children stats: count topics at `{path}/%` but NOT `{path}/%/%` (direct children only).

### Gap detection
- Query topics under a root path with `COUNT(DISTINCT nt.note_id) < min_coverage`.
- Classify as `Missing` when count = 0, `Undercovered` when 0 < count < threshold.

### Labeling algorithm
1. Embed all topic descriptions using the embedding provider.
2. For each batch of notes, embed their `normalized_text`.
3. Compute cosine similarity between each note vector and each topic vector.
4. For each note, take top-N topics above `min_confidence`.
5. Upsert into `note_topics` table with `ON CONFLICT (note_id, topic_id) DO UPDATE`.

### Cosine similarity
```
dot = sum(a[i] * b[i])
norm_a = sqrt(sum(a[i]^2))
norm_b = sqrt(sum(b[i]^2))
similarity = dot / (norm_a * norm_b)
```

### Duplicate clustering (union-find)
1. For each note, call `find_similar_to_note` via `VectorRepository` with `min_score = threshold`.
2. Collect unique pairs `(min(a,b), max(a,b))` to deduplicate.
3. Union-find with path compression. Always use smaller ID as root for determinism.
4. Group members by root, enrich with note details from PostgreSQL (text truncated to 200 chars).
5. Sort clusters by size descending.

## Acceptance Criteria

- [ ] `cargo test -p analytics` passes
- [ ] `cargo clippy -p analytics -- -D warnings` clean
- [ ] All public types are `Send + Sync`
- [ ] `Topic::parent_path()` returns `None` for root, `Some("a")` for `"a/b"`
- [ ] `Topic::depth()` returns 0 for root, 2 for `"a/b/c"`
- [ ] `Taxonomy::subtree("a")` includes `"a"`, `"a/b"`, `"a/b/c"` but not `"ab"`
- [ ] `load_taxonomy_from_yaml` handles missing file (returns empty taxonomy)
- [ ] `load_taxonomy_from_yaml` handles empty/invalid YAML (returns empty taxonomy)
- [ ] `cosine_similarity` of a vector with itself returns ~1.0
- [ ] `cosine_similarity` of orthogonal vectors returns ~0.0
- [ ] Union-find clustering deterministically uses smaller ID as representative
- [ ] `DuplicateCluster::size()` = 1 + duplicates.len()
- [ ] `GapType::Missing` when note_count = 0, `Undercovered` when 0 < count < threshold
- [ ] `AnalyticsService` is generic over `EmbeddingProvider + VectorRepository`
- [ ] Labeling respects `max_topics_per_note` limit
- [ ] Labeling skips topics below `min_confidence`

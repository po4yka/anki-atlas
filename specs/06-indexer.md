# Spec: crate `indexer`

## Source Reference

Python: `packages/indexer/` (embeddings.py, qdrant.py, service.py, service_base.py)

## Purpose

Provides text embedding via multiple providers (OpenAI, Google, local, mock) and manages the Qdrant vector database lifecycle -- upserting, deleting, and searching dense+sparse vectors for Anki notes. The `IndexService` orchestrates embedding and storage, tracking content hashes to skip unchanged notes and supporting batch operations from PostgreSQL.

## Dependencies

```toml
[dependencies]
common = { path = "../common" }
database = { path = "../database" }
async-trait = "0.1"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
reqwest = { version = "0.12", features = ["json"] }
qdrant-client = "1"
sha2 = "0.10"
blake2 = "0.10"
thiserror = "2"
tracing = "0.1"
regex = "1"

[dev-dependencies]
mockall = "0.13"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Public API

### Embeddings (`src/embeddings.rs`)

```rust
use async_trait::async_trait;

/// Errors from embedding operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("provider not configured: {0}")]
    NotConfigured(String),
    #[error("batch embedding failed: {source}")]
    BatchFailed { source: Box<dyn std::error::Error + Send + Sync> },
    #[error("rate limited, retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },
}

/// Trait for embedding providers. All impls must be Send + Sync.
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait EmbeddingProvider: Send + Sync {
    /// Model identifier for version tracking (e.g. "openai/text-embedding-3-small").
    fn model_name(&self) -> &str;

    /// Dimensionality of output vectors.
    fn dimension(&self) -> usize;

    /// Embed a batch of texts. Returns one vector per input text.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Embed a single text. Default delegates to `embed`.
    async fn embed_single(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let results = self.embed(&[text.to_owned()]).await?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    /// SHA-256[:16] hash of "{model_name}:{text}" for change detection.
    fn content_hash(&self, text: &str) -> String {
        use sha2::{Sha256, Digest};
        let input = format!("{}:{}", self.model_name(), text);
        let hash = Sha256::digest(input.as_bytes());
        hex::encode(&hash[..8]) // 16 hex chars = 8 bytes
    }
}
```

```rust
/// OpenAI embedding provider. Calls /v1/embeddings via reqwest.
pub struct OpenAiEmbeddingProvider {
    model: String,
    dimension: usize,
    batch_size: usize,
    client: reqwest::Client,
    api_key: String,
}

impl OpenAiEmbeddingProvider {
    pub fn new(model: impl Into<String>, dimension: usize, batch_size: usize) -> Result<Self, EmbeddingError>;
}
```

```rust
/// Google Gemini embedding provider.
pub struct GoogleEmbeddingProvider {
    model: String,
    dimension: usize,
    batch_size: usize,
    client: reqwest::Client,
    api_key: String,
}

impl GoogleEmbeddingProvider {
    pub fn new(model: impl Into<String>, dimension: usize, batch_size: usize) -> Result<Self, EmbeddingError>;
}
```

```rust
/// Mock embedding provider for tests. Returns deterministic vectors from MD5 hash.
#[derive(Debug, Clone)]
pub struct MockEmbeddingProvider {
    dimension: usize,
}

impl MockEmbeddingProvider {
    pub fn new(dimension: usize) -> Self;
}
```

```rust
/// Factory: create provider from config enum.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EmbeddingProviderConfig {
    OpenAi { model: String, dimension: usize, batch_size: Option<usize> },
    Google { model: String, dimension: usize, batch_size: Option<usize> },
    Mock { dimension: usize },
}

pub fn create_embedding_provider(
    config: &EmbeddingProviderConfig,
) -> Result<Box<dyn EmbeddingProvider>, EmbeddingError>;
```

### Qdrant Repository (`src/qdrant.rs`)

```rust
use serde::{Deserialize, Serialize};

/// Payload stored with each Qdrant point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NotePayload {
    pub note_id: i64,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
    pub model_id: i64,
    pub content_hash: String,
    #[serde(default)]
    pub mature: bool,
    #[serde(default)]
    pub lapses: i32,
    #[serde(default)]
    pub reps: i32,
    #[serde(default)]
    pub fail_rate: Option<f64>,
}

/// Sparse vector (indices + values) for BM25-style retrieval.
#[derive(Debug, Clone, Default)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// Errors from Qdrant operations.
#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("dimension mismatch: collection {collection} expects {expected}, got {actual}")]
    DimensionMismatch { collection: String, expected: usize, actual: usize },
    #[error("qdrant error: {0}")]
    Client(String),
    #[error("connection failed: {0}")]
    Connection(String),
}

/// Result of an upsert batch.
#[derive(Debug, Clone, Default)]
pub struct UpsertResult {
    pub upserted: usize,
    pub skipped: usize,
}

/// Trait for vector store operations. Enables mocking in tests.
#[async_trait::async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait VectorRepository: Send + Sync {
    /// Ensure collection exists with the given dense vector dimension.
    /// Returns true if newly created, false if already existed.
    async fn ensure_collection(&self, dimension: usize) -> Result<bool, VectorStoreError>;

    /// Upsert dense vectors + payloads. Optional sparse vectors.
    async fn upsert_vectors(
        &self,
        vectors: &[Vec<f32>],
        payloads: &[NotePayload],
        sparse_vectors: Option<&[SparseVector]>,
    ) -> Result<usize, VectorStoreError>;

    /// Delete points by note IDs.
    async fn delete_vectors(&self, note_ids: &[i64]) -> Result<usize, VectorStoreError>;

    /// Get content hashes for existing note IDs. Returns note_id -> hash.
    async fn get_existing_hashes(&self, note_ids: &[i64]) -> Result<std::collections::HashMap<i64, String>, VectorStoreError>;

    /// Semantic search. Returns (note_id, score) pairs.
    async fn search(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<(i64, f32)>, VectorStoreError>;

    /// Find notes similar to a given note.
    async fn find_similar_to_note(
        &self,
        note_id: i64,
        limit: usize,
        min_score: f32,
        deck_names: Option<&[String]>,
        tags: Option<&[String]>,
    ) -> Result<Vec<(i64, f32)>, VectorStoreError>;

    /// Close connection / cleanup.
    async fn close(&self) -> Result<(), VectorStoreError>;
}

/// Search filters passed to Qdrant.
#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    pub deck_names: Option<Vec<String>>,
    pub deck_names_exclude: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub tags_exclude: Option<Vec<String>>,
    pub model_ids: Option<Vec<i64>>,
    pub mature_only: bool,
    pub max_lapses: Option<i32>,
    pub min_reps: Option<i32>,
}

/// Concrete Qdrant implementation.
pub struct QdrantRepository {
    url: String,
    collection_name: String,
    // ... internal client
}

impl QdrantRepository {
    pub async fn new(url: &str, collection_name: &str) -> Result<Self, VectorStoreError>;

    /// Convert text into a hashed sparse vector (blake2b tokens, L2-normalized TF weights).
    pub fn text_to_sparse_vector(text: &str) -> SparseVector;
}
```

### Index Service (`src/service.rs`)

```rust
use serde::{Deserialize, Serialize};

/// A note prepared for indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteForIndexing {
    pub note_id: i64,
    pub model_id: i64,
    pub normalized_text: String,
    pub tags: Vec<String>,
    pub deck_names: Vec<String>,
    #[serde(default)]
    pub mature: bool,
    #[serde(default)]
    pub lapses: i32,
    #[serde(default)]
    pub reps: i32,
    #[serde(default)]
    pub fail_rate: Option<f64>,
}

/// Statistics from an indexing operation.
#[derive(Debug, Clone, Default, Serialize)]
pub struct IndexStats {
    pub notes_processed: usize,
    pub notes_embedded: usize,
    pub notes_skipped: usize,
    pub notes_deleted: usize,
    pub errors: Vec<String>,
}

/// Index service errors.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("embedding failed: {0}")]
    Embedding(#[from] crate::embeddings::EmbeddingError),
    #[error("vector store: {0}")]
    VectorStore(#[from] crate::qdrant::VectorStoreError),
    #[error("database: {0}")]
    Database(String),
    #[error("embedding model changed: stored={stored}, current={current}")]
    ModelChanged { stored: String, current: String },
}

/// Index service. Generic over dependencies for testability.
pub struct IndexService<E: EmbeddingProvider, V: VectorRepository> {
    embedding: E,
    vector_repo: V,
    db: sqlx::PgPool,
}

impl<E: EmbeddingProvider, V: VectorRepository> IndexService<E, V> {
    pub fn new(embedding: E, vector_repo: V, db: sqlx::PgPool) -> Self;

    /// Index a batch of notes. Skips notes whose content_hash is unchanged
    /// unless `force_reindex` is true.
    pub async fn index_notes(
        &self,
        notes: &[NoteForIndexing],
        force_reindex: bool,
    ) -> Result<IndexStats, IndexError>;

    /// Delete notes from the vector store by ID.
    pub async fn delete_notes(&self, note_ids: &[i64]) -> Result<usize, IndexError>;

    /// Index all notes from the PostgreSQL database.
    pub async fn index_from_database(
        &self,
        force_reindex: bool,
        batch_size: usize,
    ) -> Result<IndexStats, IndexError>;
}
```

## Internal Details

### Sparse vector construction (`text_to_sparse_vector`)
1. Lowercase the input, tokenize with regex `[a-z0-9]+`.
2. Count token frequencies.
3. For each token, hash with blake2b (4-byte digest) to get a u32 index.
4. Weight = `1.0 + ln(count)`.
5. Accumulate weights per index (hash collisions add).
6. L2-normalize the weight vector.
7. Sort by index ascending before returning.

### Content hash change detection
- Hash = `sha256("{model_name}:{text}")[..16]` (16 hex chars).
- Before embedding a batch, fetch existing hashes from Qdrant via `get_existing_hashes`.
- Skip notes where stored hash == computed hash (unless `force_reindex`).

### Embedding model versioning
- Version string: `"{normalization_version}:{model_name}:{dimension}"`.
- Stored in `sync_metadata` table (key = `embedding_version`, value = JSONB string).
- On mismatch: if `force_reindex` is false, return `IndexError::ModelChanged`; if true, recreate collection.

### OpenAI provider batching
- Process texts in chunks of `batch_size` (default 100).
- Sort response by `index` to maintain input ordering.
- Retry on 429 with exponential backoff (2, 4, 8, 16s, max 5 attempts).

### Google provider rate-limit handling
- 0.5s delay between batches.
- Retry on 429 with exponential backoff (2, 4, 8, 16s, max 5 attempts).

### Mock provider determinism
- Uses MD5 of text to produce repeatable bytes.
- Repeats hash bytes to fill the requested dimension.
- Maps each byte to `[-1.0, 1.0]` via `(byte / 127.5) - 1.0`.

### Qdrant collection creation
- Dense vector: cosine distance, optional on-disk storage.
- Sparse vector: IDF modifier.
- Optional scalar/binary quantization.
- Payload indexes on: `note_id` (int), `deck_names` (keyword), `tags` (keyword), `model_id` (int), `mature` (bool).

### Hybrid prefetch search
- When sparse vectors are supported, use `query_points` with two prefetch branches (dense + sparse) fused via Qdrant-native RRF.
- When sparse is unavailable, fall back to dense-only `query_points`.

## Acceptance Criteria

- [ ] `cargo test -p indexer` passes
- [ ] `cargo clippy -p indexer -- -D warnings` clean
- [ ] All public types are `Send + Sync`
- [ ] `MockEmbeddingProvider` produces deterministic, dimension-correct vectors
- [ ] `text_to_sparse_vector("")` returns empty indices/values
- [ ] `text_to_sparse_vector` output is sorted by index and L2-normalized
- [ ] `content_hash` includes model name (different model -> different hash)
- [ ] `index_notes` skips notes with matching hashes when `force_reindex = false`
- [ ] `index_notes` re-embeds all notes when `force_reindex = true`
- [ ] `IndexService` is generic over `EmbeddingProvider + VectorRepository` traits (mockable)
- [ ] Upsert uses `note_id` as point ID for idempotent updates
- [ ] `SearchFilters` correctly maps to Qdrant must/must_not conditions
- [ ] Integration test with `MockEmbeddingProvider` + `MockVectorRepository` covers full index-search roundtrip

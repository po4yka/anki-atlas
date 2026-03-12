# Spec: crate `indexer`

## Source Reference

Current Rust implementation: `crates/indexer/`
Historical rewrite input: `packages/indexer/` (embeddings.py, qdrant.py, service.py, service_base.py)

## Purpose

Provides embedding via multiple providers (OpenAI, Google, mock) and manages the Qdrant vector database lifecycle for chunk-aware Anki note indexing. The crate now supports multimodal Gemini Embedding 2 requests, chunk-level Qdrant payloads, note-level content hash tracking, and semantic retrieval over both note-level and raw chunk results.

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
    #[error("http {status}: {body}")]
    Http { status: u16, body: String },
    #[error("unsupported input for provider: {message}")]
    UnsupportedInput { message: String },
    #[error("protocol error: {message}")]
    Protocol { message: String },
    #[error("batch embedding failed: {source}")]
    BatchFailed { source: Box<dyn std::error::Error + Send + Sync> },
    #[error("rate limited, retry after {retry_after_secs}s")]
    RateLimited { retry_after_secs: u64 },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingTask {
    #[default]
    Default,
    RetrievalDocument,
    RetrievalQuery,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EmbeddingPart {
    Text { text: String },
    InlineBytes { mime_type: String, data: Vec<u8>, display_name: Option<String> },
    FileUri { mime_type: String, uri: String, display_name: Option<String> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingInput {
    pub parts: Vec<EmbeddingPart>,
    #[serde(default)]
    pub task: EmbeddingTask,
    pub title: Option<String>,
    pub output_dimensionality: Option<usize>,
}

/// Trait for embedding providers. All impls must be Send + Sync.
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait EmbeddingProvider: Send + Sync {
    /// Model identifier for version tracking.
    fn model_name(&self) -> &str;

    /// Dimensionality of output vectors.
    fn dimension(&self) -> usize;

    /// Embed a batch of multimodal inputs. Returns one vector per input.
    async fn embed_inputs(
        &self,
        inputs: &[EmbeddingInput],
    ) -> Result<Vec<Vec<f32>>, EmbeddingError>;

    /// Embed a batch of texts. Returns one vector per input text.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
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

Implementation notes:

- `gemini-embedding-001` stays on the legacy text batch path.
- `gemini-embedding-2-preview` uses `embedContent` with multimodal `EmbeddingInput` values.
- Gemini Embedding 2 may upload large assets through the Google Files API and then reference them by file URI.
- Google provider bootstrap prefers `GEMINI_API_KEY`, with `GOOGLE_API_KEY` as a backward-compatible fallback.

```rust
/// Mock embedding provider for tests. Returns deterministic vectors from SHA-256-derived bytes.
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
    OpenAi { model: String, dimension: usize, batch_size: Option<usize>, api_key: String },
    Google { model: String, dimension: usize, batch_size: Option<usize>, api_key: String },
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
    #[serde(default)]
    pub chunk_id: String,
    #[serde(default)]
    pub chunk_kind: String,
    #[serde(default)]
    pub modality: String,
    #[serde(default)]
    pub source_field: Option<String>,
    #[serde(default)]
    pub asset_rel_path: Option<String>,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub preview_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticSearchHit {
    pub note_id: i64,
    pub chunk_id: String,
    pub chunk_kind: String,
    pub modality: String,
    pub source_field: Option<String>,
    pub asset_rel_path: Option<String>,
    pub mime_type: Option<String>,
    pub preview_label: Option<String>,
    pub score: f32,
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
    #[error("reindex required: {reason}")]
    ReindexRequired { reason: String },
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

    /// Return the current collection dimension, or `None` if the collection does not exist.
    async fn collection_dimension(&self) -> Result<Option<usize>, VectorStoreError>;

    /// Drop and recreate the collection with the requested dimension.
    async fn recreate_collection(&self, dimension: usize) -> Result<(), VectorStoreError>;

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

    /// Semantic search against chunk payloads.
    async fn search_chunks(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SemanticSearchHit>, VectorStoreError>;

    /// Semantic search aggregated to note IDs. Returns (note_id, score) pairs.
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkForIndexing {
    pub chunk_id: String,
    pub chunk_kind: String,
    pub modality: String,
    pub embedding_input: EmbeddingInput,
    pub sparse_text: Option<String>,
    pub source_field: Option<String>,
    pub asset_rel_path: Option<String>,
    pub mime_type: Option<String>,
    pub preview_label: Option<String>,
    pub hash_component: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalNoteForIndexing {
    pub note: NoteForIndexing,
    pub chunks: Vec<ChunkForIndexing>,
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
    embedding: Arc<E>,
    vector_repo: Arc<V>,
}

impl<E: EmbeddingProvider, V: VectorRepository> IndexService<E, V> {
    pub fn new(embedding: E, vector_repo: V) -> Self;

    /// Index a batch of notes. Skips notes whose content_hash is unchanged
    /// unless `force_reindex` is true.
    pub async fn index_notes(
        &self,
        notes: &[NoteForIndexing],
        force_reindex: bool,
    ) -> Result<IndexStats, IndexError>;

    /// Index a batch of multimodal notes with explicit chunk definitions.
    pub async fn index_multimodal_notes_with_progress(
        &self,
        notes: &[MultimodalNoteForIndexing],
        force_reindex: bool,
        progress: Option<IndexProgressCallback>,
    ) -> Result<IndexStats, IndexError>;

    /// Delete notes from the vector store by ID.
    pub async fn delete_notes(&self, note_ids: &[i64]) -> Result<usize, IndexError>;
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
- Hash = `sha256(model_name + all chunk hash parts)[..16]` (16 hex chars).
- Text-only notes use a default `text_primary` chunk.
- Multimodal notes hash the text chunk plus every asset hash component, so changing an image/audio/video/PDF forces note re-embedding.
- Before embedding a batch, fetch existing hashes from Qdrant via `get_existing_hashes`.
- Skip notes where stored hash == computed hash (unless `force_reindex`).

### Google provider behavior
- `gemini-embedding-001` stays on the text-only `batchEmbedContents` path.
- `gemini-embedding-2-preview` uses per-input multimodal requests, `EmbeddingTask`, optional `title`, and optional `output_dimensionality`.
- Gemini Embedding 2 can send small assets inline and upload larger assets through the Files API.
- Google retries rate limits with bounded backoff and bounded concurrency rather than one large text batch loop.

### Mock provider determinism
- Uses SHA-256 over task, title, requested output dimensionality, and all embedding parts.
- Repeats hash bytes to fill the requested dimension.
- Maps each byte to `[-1.0, 1.0]` via `(byte / 127.5) - 1.0`.

### Qdrant collection creation
- Dense vector: cosine distance, optional on-disk storage.
- Sparse vector: IDF modifier.
- Optional scalar/binary quantization.
- Payload indexes on: `note_id` (int), `deck_names` (keyword), `tags` (keyword), `model_id` (int), `mature` (bool).
- One note may own many chunk payloads; deletes operate by payload `note_id` rather than one-point-per-note IDs.
- Chunk point IDs are stable and derived from `note_id`, `chunk_kind`, and a hashed suffix.

### Collection compatibility and rollout
- Runtime stores `embedding_model`, `embedding_dimension`, and `embedding_vector_schema` in `sync_metadata`.
- Explicit index flows may recreate the collection when model, dimension, or vector schema is incompatible.
- Read-only surfaces use the same repository introspection to return `reindex required` instead of mutating Qdrant.

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
- [ ] `content_hash_parts` includes model name and all chunk hash parts
- [ ] `index_notes` skips notes with matching hashes when `force_reindex = false`
- [ ] `index_notes` re-embeds all notes when `force_reindex = true`
- [ ] multimodal note indexing produces multiple chunk payloads for a single note when supported media is present
- [ ] `IndexService` is generic over `EmbeddingProvider + VectorRepository` traits (mockable)
- [ ] delete-by-note removes all chunk payloads for the requested note IDs
- [ ] Gemini Embedding 2 accepts `EmbeddingInput` values with text, inline bytes, and file URIs
- [ ] `SearchFilters` correctly maps to Qdrant must/must_not conditions
- [ ] collection mismatch handling covers model, dimension, and vector schema compatibility

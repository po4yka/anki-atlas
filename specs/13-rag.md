# Spec: crate `rag`

## Source Reference
Current Rust implementation: `crates/rag/`
Historical rewrite input: `packages/rag/` (chunker.py, service.py, store.py)

## Purpose
Retrieval-augmented generation subsystem: chunk markdown documents into typed segments for embedding, store and query vectors via an abstract trait (backed by Qdrant in production), and provide high-level RAG operations for few-shot retrieval, context enrichment, and duplicate detection. The store trait is generic so tests can use an in-memory implementation.

## Dependencies
```toml
[dependencies]
common = { path = "../common" }
async-trait = "0.1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.10"
regex = "1"
thiserror = "2"
tracing = "0.1"
strum = { version = "0.27", features = ["derive"] }

[dev-dependencies]
mockall = "0.13"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Public API

### Error (`src/error.rs`)

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RagError {
    #[error("vector store error: {0}")]
    Store(String),

    #[error("chunking error: {0}")]
    Chunking(String),

    #[error("search error: {0}")]
    Search(String),
}
```

### Chunker (`src/chunker.rs`)

```rust
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};
use std::collections::HashMap;

/// Type of document chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, EnumString, Display)]
#[strum(serialize_all = "snake_case")]
pub enum ChunkType {
    Summary,
    KeyPoints,
    CodeExample,
    Question,
    Answer,
    FullContent,
    Section,
}

/// A chunk of document content with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub chunk_id: String,
    pub content: String,
    pub chunk_type: ChunkType,
    pub source_file: String,
    pub content_hash: String,          // first 16 hex chars of SHA-256
    pub metadata: HashMap<String, String>,
}

/// Configuration for the document chunker.
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    pub chunk_size: usize,             // default 1000
    pub min_chunk_size: usize,         // default 50
    pub include_code_blocks: bool,     // default true
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self { chunk_size: 1000, min_chunk_size: 50, include_code_blocks: true }
    }
}

/// Split markdown documents into typed chunks for embedding.
pub struct DocumentChunker {
    config: ChunkerConfig,
}

impl DocumentChunker {
    pub fn new(config: ChunkerConfig) -> Self;

    /// Parse and chunk markdown content.
    ///
    /// Extracts heading-based sections, code blocks, and falls back to
    /// full content if no structured chunks found.
    pub fn chunk_content(
        &self,
        content: &str,
        source_file: &str,
        frontmatter: Option<&HashMap<String, String>>,
    ) -> Vec<DocumentChunk>;

    /// Read and chunk a single markdown file.
    pub fn chunk_file(&self, path: &std::path::Path) -> Vec<DocumentChunk>;
}
```

### Store Trait (`src/store.rs`)

```rust
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::error::RagError;

/// Result from vector similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub content: String,
    pub score: f32,                    // raw distance
    pub source_file: String,
    pub metadata: HashMap<String, String>,
}

impl SearchResult {
    /// Convert distance to similarity (0-1): 1 / (1 + distance).
    pub fn similarity(&self) -> f32 {
        1.0 / (1.0 + self.score)
    }
}

/// Filter clause for metadata queries.
#[derive(Debug, Clone, Default)]
pub struct MetadataFilter {
    pub field: String,
    pub value: String,
}

/// Abstract vector store trait for DI.
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait VectorStore: Send + Sync {
    /// Add documents with embeddings. Returns count of newly added.
    async fn add(
        &self,
        ids: &[String],
        documents: &[String],
        embeddings: &[Vec<f32>],
        metadatas: Option<&[HashMap<String, String>]>,
    ) -> Result<usize, RagError>;

    /// Delete all chunks from a specific source file.
    async fn delete_by_source(&self, source_file: &str) -> Result<usize, RagError>;

    /// Reset (drop and recreate) the collection.
    async fn reset(&self) -> Result<(), RagError>;

    /// Search by embedding vector.
    async fn search(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&MetadataFilter>,
        min_similarity: f32,
    ) -> Result<Vec<SearchResult>, RagError>;

    /// Return total document count.
    async fn count(&self) -> Result<usize, RagError>;

    /// Return store statistics.
    async fn get_stats(&self) -> Result<StoreStats, RagError>;
}

/// Store-level statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreStats {
    pub total_chunks: usize,
    pub unique_files: usize,
    pub topics: Vec<String>,
}
```

### RAG Service (`src/service.rs`)

```rust
use crate::store::{SearchResult, VectorStore, MetadataFilter};
use crate::error::RagError;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A related concept from the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedConcept {
    pub title: String,
    pub content: String,
    pub topic: String,
    pub similarity: f32,
    pub source_file: String,
}

/// Result of a duplicate detection check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateCheckResult {
    pub is_duplicate: bool,
    pub confidence: f32,
    pub similar_items: Vec<SearchResult>,
    pub recommendation: String,
}

/// A few-shot example for card generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotExample {
    pub question: String,
    pub answer: String,
    pub topic: String,
    pub difficulty: String,
    pub source_file: String,
}

/// High-level RAG operations for flashcard generation.
/// Accepts pre-computed embedding vectors (caller controls embedding model).
pub struct RagService<S: VectorStore> {
    store: Arc<S>,
}

impl<S: VectorStore> RagService<S> {
    pub fn new(store: Arc<S>) -> Self;

    /// Check whether a card is a potential duplicate.
    pub async fn find_duplicates(
        &self,
        query_embedding: &[f32],
        threshold: f32,        // default 0.85
        k: usize,             // default 5
    ) -> Result<DuplicateCheckResult, RagError>;

    /// Retrieve related concepts for context enrichment.
    /// De-duplicates by source_file.
    pub async fn get_context(
        &self,
        query_embedding: &[f32],
        k: usize,             // default 5
        topic: Option<&str>,
        min_similarity: f32,   // default 0.3
    ) -> Result<Vec<RelatedConcept>, RagError>;

    /// Retrieve few-shot examples for generation prompts.
    pub async fn get_few_shot_examples(
        &self,
        query_embedding: &[f32],
        k: usize,             // default 3
        topic: Option<&str>,
    ) -> Result<Vec<FewShotExample>, RagError>;
}
```

### Module root (`src/lib.rs`)

```rust
pub mod chunker;
pub mod error;
pub mod service;
pub mod store;

pub use chunker::{ChunkType, ChunkerConfig, DocumentChunk, DocumentChunker};
pub use error::RagError;
pub use service::{DuplicateCheckResult, FewShotExample, RagService, RelatedConcept};
pub use store::{MetadataFilter, SearchResult, StoreStats, VectorStore};
```

## Internal Details

### Chunk ID Generation
- `{file_stem}_{section_name}_{path_hash_8chars}` where `path_hash` is first 8 hex chars of SHA-256 of the posix path.

### Content Hash
- First 16 hex characters of SHA-256 of the chunk content.

### Section Classification
- Heading text is matched case-insensitively:
  - Contains "summary" -> `Summary`
  - Contains "key point" -> `KeyPoints`
  - Contains "question" -> `Question`
  - Contains "answer" -> `Answer`
  - Default -> `Section`

### Truncation
- Content exceeding `chunk_size` is truncated at the last space boundary in the first half, appending `"..."`.

### Duplicate Recommendation Thresholds
- >= 0.95: "Highly likely duplicate -- skip this card"
- >= 0.85: "Probable duplicate -- review before creating"
- >= 0.70: "Similar content exists -- consider differentiating"
- < 0.70: "No significant duplicates found"

### Context De-duplication
- `get_context` fetches `k*2` results, then de-duplicates by `source_file`, keeping first occurrence, up to `k` results.

### Few-Shot Examples
- `get_few_shot_examples` fetches `k*3` results, truncates `question` to 300 chars, `answer` to 500 chars, returns up to `k`.

## Acceptance Criteria
- [ ] `DocumentChunker::chunk_content` splits by heading into typed sections
- [ ] `DocumentChunker::chunk_content` extracts code blocks as `CodeExample` chunks
- [ ] `DocumentChunker::chunk_content` falls back to `FullContent` when no structure
- [ ] `DocumentChunker::chunk_content` skips chunks smaller than `min_chunk_size`
- [ ] `DocumentChunker::chunk_content` truncates chunks exceeding `chunk_size` at word boundary
- [ ] Chunk IDs are deterministic for the same input
- [ ] Content hashes are deterministic SHA-256 prefix
- [ ] Section classification maps "summary", "key point", "question", "answer" correctly
- [ ] `SearchResult::similarity()` computes `1/(1+score)` correctly
- [ ] `MockVectorStore` compiles and can be used in tests
- [ ] `RagService::find_duplicates` returns `is_duplicate=true` when results exist above threshold
- [ ] `RagService::find_duplicates` returns correct recommendation string per confidence level
- [ ] `RagService::get_context` de-duplicates by source_file
- [ ] `RagService::get_context` respects topic filter
- [ ] `RagService::get_context` returns at most `k` results
- [ ] `RagService::get_few_shot_examples` truncates question/answer fields
- [ ] `VectorStore::add` returns count of newly added (skips existing IDs)
- [ ] All types are `Send + Sync` (compile-time assertion)
- [ ] `make check` equivalent passes (clippy, fmt, test)

mod client;
mod repository;
mod schema;

pub use client::QdrantRepository;
#[cfg(test)]
pub use repository::MockVectorRepository;
pub use repository::VectorRepository;
pub use schema::{
    NotePayload, ScoredNote, SearchFilters, SemanticSearchHit, SparseVector, UpsertResult,
    VectorStoreError,
};

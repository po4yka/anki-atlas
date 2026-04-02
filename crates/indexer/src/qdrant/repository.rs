use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use super::schema::{
    NotePayload, ScoredNote, SearchFilters, SemanticSearchHit, SparseVector, VectorStoreError,
};

/// Trait for vector store operations. Enables mocking in tests.
#[async_trait]
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
    async fn get_existing_hashes(
        &self,
        note_ids: &[i64],
    ) -> Result<HashMap<i64, String>, VectorStoreError>;

    /// Semantic search against chunk vectors.
    async fn search_chunks(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SemanticSearchHit>, VectorStoreError>;

    /// Semantic search. Returns scored notes sorted by descending score.
    async fn search(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<ScoredNote>, VectorStoreError> {
        let mut by_note = HashMap::<i64, f32>::new();
        for hit in self
            .search_chunks(
                query_vector,
                query_sparse,
                limit.saturating_mul(4).max(limit),
                filters,
            )
            .await?
        {
            by_note
                .entry(hit.note_id)
                .and_modify(|score| {
                    if hit.score > *score {
                        *score = hit.score;
                    }
                })
                .or_insert(hit.score);
        }
        let mut results: Vec<ScoredNote> = by_note
            .into_iter()
            .map(|(note_id, score)| ScoredNote { note_id, score })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        Ok(results)
    }

    /// Find notes similar to a given note.
    async fn find_similar_to_note(
        &self,
        note_id: i64,
        limit: usize,
        min_score: f32,
        deck_names: Option<&[String]>,
        tags: Option<&[String]>,
    ) -> Result<Vec<ScoredNote>, VectorStoreError>;

    /// Close connection / cleanup.
    async fn close(&self) -> Result<(), VectorStoreError>;
}

#[async_trait]
impl<T> VectorRepository for &T
where
    T: VectorRepository + ?Sized,
{
    async fn ensure_collection(&self, dimension: usize) -> Result<bool, VectorStoreError> {
        (*self).ensure_collection(dimension).await
    }

    async fn collection_dimension(&self) -> Result<Option<usize>, VectorStoreError> {
        (*self).collection_dimension().await
    }

    async fn recreate_collection(&self, dimension: usize) -> Result<(), VectorStoreError> {
        (*self).recreate_collection(dimension).await
    }

    async fn upsert_vectors(
        &self,
        vectors: &[Vec<f32>],
        payloads: &[NotePayload],
        sparse_vectors: Option<&[SparseVector]>,
    ) -> Result<usize, VectorStoreError> {
        (*self)
            .upsert_vectors(vectors, payloads, sparse_vectors)
            .await
    }

    async fn delete_vectors(&self, note_ids: &[i64]) -> Result<usize, VectorStoreError> {
        (*self).delete_vectors(note_ids).await
    }

    async fn get_existing_hashes(
        &self,
        note_ids: &[i64],
    ) -> Result<HashMap<i64, String>, VectorStoreError> {
        (*self).get_existing_hashes(note_ids).await
    }

    async fn search_chunks(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SemanticSearchHit>, VectorStoreError> {
        (*self)
            .search_chunks(query_vector, query_sparse, limit, filters)
            .await
    }

    async fn search(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<ScoredNote>, VectorStoreError> {
        (*self)
            .search(query_vector, query_sparse, limit, filters)
            .await
    }

    async fn find_similar_to_note(
        &self,
        note_id: i64,
        limit: usize,
        min_score: f32,
        deck_names: Option<&[String]>,
        tags: Option<&[String]>,
    ) -> Result<Vec<ScoredNote>, VectorStoreError> {
        (*self)
            .find_similar_to_note(note_id, limit, min_score, deck_names, tags)
            .await
    }

    async fn close(&self) -> Result<(), VectorStoreError> {
        (*self).close().await
    }
}

#[async_trait]
impl<T> VectorRepository for Box<T>
where
    T: VectorRepository + ?Sized,
{
    async fn ensure_collection(&self, dimension: usize) -> Result<bool, VectorStoreError> {
        (**self).ensure_collection(dimension).await
    }

    async fn collection_dimension(&self) -> Result<Option<usize>, VectorStoreError> {
        (**self).collection_dimension().await
    }

    async fn recreate_collection(&self, dimension: usize) -> Result<(), VectorStoreError> {
        (**self).recreate_collection(dimension).await
    }

    async fn upsert_vectors(
        &self,
        vectors: &[Vec<f32>],
        payloads: &[NotePayload],
        sparse_vectors: Option<&[SparseVector]>,
    ) -> Result<usize, VectorStoreError> {
        (**self)
            .upsert_vectors(vectors, payloads, sparse_vectors)
            .await
    }

    async fn delete_vectors(&self, note_ids: &[i64]) -> Result<usize, VectorStoreError> {
        (**self).delete_vectors(note_ids).await
    }

    async fn get_existing_hashes(
        &self,
        note_ids: &[i64],
    ) -> Result<HashMap<i64, String>, VectorStoreError> {
        (**self).get_existing_hashes(note_ids).await
    }

    async fn search_chunks(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SemanticSearchHit>, VectorStoreError> {
        (**self)
            .search_chunks(query_vector, query_sparse, limit, filters)
            .await
    }

    async fn search(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<ScoredNote>, VectorStoreError> {
        (**self)
            .search(query_vector, query_sparse, limit, filters)
            .await
    }

    async fn find_similar_to_note(
        &self,
        note_id: i64,
        limit: usize,
        min_score: f32,
        deck_names: Option<&[String]>,
        tags: Option<&[String]>,
    ) -> Result<Vec<ScoredNote>, VectorStoreError> {
        (**self)
            .find_similar_to_note(note_id, limit, min_score, deck_names, tags)
            .await
    }

    async fn close(&self) -> Result<(), VectorStoreError> {
        (**self).close().await
    }
}

#[async_trait]
impl<T> VectorRepository for Arc<T>
where
    T: VectorRepository + ?Sized,
{
    async fn ensure_collection(&self, dimension: usize) -> Result<bool, VectorStoreError> {
        (**self).ensure_collection(dimension).await
    }

    async fn collection_dimension(&self) -> Result<Option<usize>, VectorStoreError> {
        (**self).collection_dimension().await
    }

    async fn recreate_collection(&self, dimension: usize) -> Result<(), VectorStoreError> {
        (**self).recreate_collection(dimension).await
    }

    async fn upsert_vectors(
        &self,
        vectors: &[Vec<f32>],
        payloads: &[NotePayload],
        sparse_vectors: Option<&[SparseVector]>,
    ) -> Result<usize, VectorStoreError> {
        (**self)
            .upsert_vectors(vectors, payloads, sparse_vectors)
            .await
    }

    async fn delete_vectors(&self, note_ids: &[i64]) -> Result<usize, VectorStoreError> {
        (**self).delete_vectors(note_ids).await
    }

    async fn get_existing_hashes(
        &self,
        note_ids: &[i64],
    ) -> Result<HashMap<i64, String>, VectorStoreError> {
        (**self).get_existing_hashes(note_ids).await
    }

    async fn search_chunks(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<SemanticSearchHit>, VectorStoreError> {
        (**self)
            .search_chunks(query_vector, query_sparse, limit, filters)
            .await
    }

    async fn search(
        &self,
        query_vector: &[f32],
        query_sparse: Option<&SparseVector>,
        limit: usize,
        filters: &SearchFilters,
    ) -> Result<Vec<ScoredNote>, VectorStoreError> {
        (**self)
            .search(query_vector, query_sparse, limit, filters)
            .await
    }

    async fn find_similar_to_note(
        &self,
        note_id: i64,
        limit: usize,
        min_score: f32,
        deck_names: Option<&[String]>,
        tags: Option<&[String]>,
    ) -> Result<Vec<ScoredNote>, VectorStoreError> {
        (**self)
            .find_similar_to_note(note_id, limit, min_score, deck_names, tags)
            .await
    }

    async fn close(&self) -> Result<(), VectorStoreError> {
        (**self).close().await
    }
}

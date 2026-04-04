use indexer::embeddings::{EmbeddingInput, EmbeddingProvider, EmbeddingTask};
use indexer::qdrant::{
    QdrantRepository, SearchFilters as VectorFilters, SemanticSearchHit, VectorRepository,
};

use crate::error::SearchError;
use crate::fts::SearchFilters;

/// Convert FTS `SearchFilters` into the vector-store filter type.
pub(crate) fn to_vector_filters(filters: Option<&SearchFilters>) -> VectorFilters {
    let Some(filters) = filters else {
        return VectorFilters::default();
    };

    VectorFilters {
        deck_names: filters.deck_names.clone(),
        deck_names_exclude: filters.deck_names_exclude.clone(),
        tags: filters.tags.clone(),
        tags_exclude: filters.tags_exclude.clone(),
        model_ids: filters.model_ids.clone(),
        mature_only: filters.min_ivl.is_some_and(|min_ivl| min_ivl >= 21),
        max_lapses: filters.max_lapses,
        min_reps: filters.min_reps,
    }
}

/// Embed `query` and run a chunk-level vector search, returning raw hits.
///
/// When the query is non-empty, also generates a sparse query vector for
/// Qdrant-native hybrid search (dense + sparse with RRF fusion).
pub(crate) async fn run_semantic_chunk_search<E, V>(
    embedding: &E,
    vector_repo: &V,
    query: &str,
    filters: Option<&SearchFilters>,
    limit: usize,
) -> Result<Vec<SemanticSearchHit>, SearchError>
where
    E: EmbeddingProvider,
    V: VectorRepository,
{
    let embedded = embedding
        .embed_inputs(&[EmbeddingInput::text_with_task(
            query.to_string(),
            EmbeddingTask::RetrievalQuery,
        )])
        .await?;
    let query_vector = &embedded[0];
    let vector_filters = to_vector_filters(filters);

    // Generate sparse query vector for hybrid search
    let sparse = QdrantRepository::text_to_sparse_vector(query);
    let sparse_ref = if sparse.indices.is_empty() {
        None
    } else {
        Some(&sparse)
    };

    vector_repo
        .search_chunks(
            query_vector,
            sparse_ref,
            limit.saturating_mul(4).max(limit),
            &vector_filters,
        )
        .await
        .map_err(Into::into)
}

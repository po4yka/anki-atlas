use std::collections::HashMap;

use tracing::instrument;

use crate::error::SearchError;
use crate::reranker::Reranker;

use super::SearchService;
use super::types::{ChunkSearchHit, ChunkSearchParams, ChunkSearchResult, NoteDetail};

impl<E, V, R> SearchService<E, V, R>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
    R: Reranker,
{
    /// Execute semantic-only chunk search.
    #[instrument(skip(self))]
    pub async fn search_chunks(
        &self,
        params: &ChunkSearchParams,
    ) -> Result<ChunkSearchResult, SearchError> {
        if params.query.trim().is_empty() {
            return Ok(ChunkSearchResult {
                query: params.query.clone(),
                results: Vec::new(),
            });
        }

        let raw = crate::semantic::run_semantic_chunk_search(
            &self.embedding,
            &self.vector_repo,
            &params.query,
            params.filters.as_ref(),
            params.limit,
        )
        .await?;
        let mut results: Vec<_> = raw
            .into_iter()
            .map(|hit| ChunkSearchHit {
                note_id: hit.note_id,
                chunk_id: hit.chunk_id,
                chunk_kind: hit.chunk_kind,
                modality: hit.modality,
                source_field: hit.source_field,
                asset_rel_path: hit.asset_rel_path,
                mime_type: hit.mime_type,
                preview_label: hit.preview_label,
                score: f64::from(hit.score),
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(params.limit);

        Ok(ChunkSearchResult {
            query: params.query.clone(),
            results,
        })
    }

    /// Fetch note details for a list of IDs (for reranking / enrichment).
    #[instrument(skip(self))]
    pub async fn get_notes_details(
        &self,
        note_ids: &[i64],
    ) -> Result<HashMap<i64, NoteDetail>, SearchError> {
        self.repository.get_note_details(note_ids).await
    }
}

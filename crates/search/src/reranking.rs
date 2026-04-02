use std::collections::HashMap;
use std::sync::Arc;

use crate::error::SearchError;
use crate::fusion::SearchResult;
use crate::repository::SearchReadRepository;
use crate::reranker::Reranker;
use crate::service::NoteDetail;

/// Fetch the top-N results' note texts for reranking.
///
/// Returns `(note_id, normalized_text)` pairs; results with empty text are
/// dropped so the reranker is never handed empty documents.
pub(crate) async fn build_rerank_documents(
    results: &[SearchResult],
    rerank_top_n: usize,
    repository: &Arc<dyn SearchReadRepository>,
) -> Result<Vec<(i64, String)>, SearchError> {
    let top_note_ids: Vec<i64> = results
        .iter()
        .take(rerank_top_n)
        .map(|r| r.note_id)
        .collect();

    let note_details: HashMap<i64, NoteDetail> = repository.get_note_details(&top_note_ids).await?;

    Ok(results
        .iter()
        .take(rerank_top_n)
        .filter_map(|result| {
            let note_text = note_details
                .get(&result.note_id)
                .map(|detail| detail.normalized_text.clone())
                .or_else(|| result.headline.clone())?;

            if note_text.trim().is_empty() {
                None
            } else {
                Some((result.note_id, note_text))
            }
        })
        .collect())
}

/// Optionally rerank `results` in-place using `reranker`.
///
/// Returns `(rerank_applied, rerank_model)`. On reranker error the function
/// logs a warning and leaves results in their original RRF order.
pub(crate) async fn apply_reranking<R: Reranker>(
    results: &mut [SearchResult],
    query: &str,
    reranker: &R,
    rerank_top_n: usize,
    repository: &Arc<dyn SearchReadRepository>,
) -> (bool, Option<String>) {
    match build_rerank_documents(results, rerank_top_n, repository).await {
        Ok(documents) if !documents.is_empty() => match reranker.rerank(query, &documents).await {
            Ok(scores) => {
                let score_map: HashMap<i64, f64> = scores.into_iter().collect();
                for result in results.iter_mut() {
                    if let Some(&score) = score_map.get(&result.note_id) {
                        result.rerank_score = Some(score);
                    }
                }
                results.sort_by(|a, b| {
                    b.rerank_score
                        .partial_cmp(&a.rerank_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                (true, Some(reranker.model_name().to_string()))
            }
            Err(e) => {
                tracing::warn!(error = %e, "reranking failed, falling back to RRF ordering");
                (false, None)
            }
        },
        Err(e) => {
            tracing::warn!(
                error = %e,
                "failed to build rerank documents, skipping reranking"
            );
            (false, None)
        }
        _ => (false, None),
    }
}

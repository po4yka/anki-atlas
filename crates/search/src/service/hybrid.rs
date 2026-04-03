use std::collections::HashMap;

use tracing::instrument;

use crate::error::SearchError;
use crate::fts::LexicalMode;
use crate::reranker::{Reranker, ScoredNote};

use super::SearchService;
use super::types::{HybridSearchResult, SearchMode, SearchParams};

impl<E, V, R> SearchService<E, V, R>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
    R: Reranker,
{
    /// Execute hybrid search: semantic + FTS -> RRF fusion -> optional rerank.
    #[instrument(skip(self))]
    pub async fn search(&self, params: &SearchParams) -> Result<HybridSearchResult, SearchError> {
        let SearchParams {
            ref query,
            filters: ref _filters,
            limit,
            semantic_weight,
            fts_weight,
            search_mode,
            rerank_override,
            rerank_top_n_override,
        } = *params;

        // Empty/whitespace query short-circuit
        if query.trim().is_empty() {
            return Ok(HybridSearchResult::empty(query.to_string()));
        }

        // Semantic search
        let mut semantic_matches = HashMap::<i64, indexer::qdrant::SemanticSearchHit>::new();
        let semantic_results = if search_mode == SearchMode::FtsOnly {
            vec![]
        } else {
            let raw = crate::semantic::run_semantic_chunk_search(
                &self.embedding,
                &self.vector_repo,
                query,
                params.filters.as_ref(),
                limit,
            )
            .await?;
            let mut best_by_note = HashMap::<i64, f64>::new();
            for hit in raw {
                let score = f64::from(hit.score);
                best_by_note
                    .entry(hit.note_id)
                    .and_modify(|best_score| {
                        if score > *best_score {
                            *best_score = score;
                        }
                    })
                    .or_insert(score);
                semantic_matches
                    .entry(hit.note_id)
                    .and_modify(|existing| {
                        if hit.score > existing.score {
                            *existing = hit.clone();
                        }
                    })
                    .or_insert(hit);
            }
            let mut semantic_results: Vec<ScoredNote> = best_by_note
                .into_iter()
                .map(|(note_id, score)| ScoredNote { note_id, score })
                .collect();
            semantic_results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            semantic_results.truncate(limit);
            semantic_results
        };

        // FTS search
        let (
            fts_results,
            lexical_mode,
            lexical_fallback_used,
            query_suggestions,
            autocomplete_suggestions,
        ) = if search_mode == SearchMode::SemanticOnly {
            (vec![], LexicalMode::None, false, vec![], vec![])
        } else {
            let lexical = self
                .repository
                .search_lexical(query, params.filters.as_ref(), limit as i64)
                .await?;
            let fts_results = lexical
                .results
                .into_iter()
                .map(|r| (r.note_id, r.rank, r.headline))
                .collect();
            (
                fts_results,
                lexical.mode,
                lexical.used_fallback,
                lexical.query_suggestions,
                lexical.autocomplete_suggestions,
            )
        };

        // RRF fusion
        let (mut results, stats) = crate::fusion::reciprocal_rank_fusion(
            &semantic_results,
            &fts_results,
            60,
            limit,
            if search_mode == SearchMode::FtsOnly {
                0.0
            } else {
                semantic_weight
            },
            if search_mode == SearchMode::SemanticOnly {
                0.0
            } else {
                fts_weight
            },
        );

        for result in &mut results {
            if let Some(hit) = semantic_matches.get(&result.note_id) {
                result.match_modality = Some(hit.modality.clone());
                result.match_chunk_kind = Some(hit.chunk_kind.clone());
                result.match_source_field = hit.source_field.clone();
                result.match_asset_rel_path = hit.asset_rel_path.clone();
                result.match_preview_label = hit.preview_label.clone();
            }
        }

        // Determine whether to rerank
        let should_rerank = rerank_override.unwrap_or(self.rerank_enabled);
        let rerank_top_n = rerank_top_n_override.unwrap_or(self.rerank_top_n);
        let mut rerank_applied = false;
        let mut rerank_model: Option<String> = None;

        if should_rerank {
            if let Some(ref reranker) = self.reranker {
                (rerank_applied, rerank_model) = crate::reranking::apply_reranking(
                    &mut results,
                    query,
                    reranker,
                    rerank_top_n,
                    &self.repository,
                )
                .await;
            }
            // No reranker provided: degrade gracefully
        }

        // Apply limit
        results.truncate(limit);

        Ok(HybridSearchResult {
            results,
            stats,
            query: query.to_string(),
            filters_applied: HashMap::new(),
            lexical_mode,
            lexical_fallback_used,
            query_suggestions,
            autocomplete_suggestions,
            rerank_applied,
            rerank_model: if rerank_applied { rerank_model } else { None },
            rerank_top_n: if rerank_applied {
                Some(rerank_top_n)
            } else {
                None
            },
        })
    }
}

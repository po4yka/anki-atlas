use std::collections::HashMap;

use serde::Serialize;

/// Fused search result with score breakdown from all retrieval stages.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub note_id: i64,
    pub rrf_score: f64,
    pub semantic_score: Option<f64>,
    pub semantic_rank: Option<usize>,
    pub fts_score: Option<f64>,
    pub fts_rank: Option<usize>,
    pub headline: Option<String>,
    pub rerank_score: Option<f64>,
    pub rerank_rank: Option<usize>,
}

impl SearchResult {
    /// Return list of contributing sources ("semantic", "fts").
    pub fn sources(&self) -> Vec<&'static str> {
        let mut sources = Vec::new();
        if self.semantic_score.is_some() {
            sources.push("semantic");
        }
        if self.fts_score.is_some() {
            sources.push("fts");
        }
        sources
    }
}

/// Statistics about the fusion operation.
#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct FusionStats {
    pub semantic_only: usize,
    pub fts_only: usize,
    pub both: usize,
    pub total: usize,
}

/// Fuse semantic and FTS results using Reciprocal Rank Fusion.
///
/// RRF score = sum( weight / (k + rank) ) across sources.
pub fn reciprocal_rank_fusion(
    semantic_results: &[(i64, f64)],
    fts_results: &[(i64, f64, Option<String>)],
    k: usize,
    limit: usize,
    semantic_weight: f64,
    fts_weight: f64,
) -> (Vec<SearchResult>, FusionStats) {
    let mut entries: HashMap<i64, SearchResult> = HashMap::new();

    for (rank_idx, &(note_id, score)) in semantic_results.iter().enumerate() {
        let rank = rank_idx + 1; // 1-indexed
        let rrf_score = semantic_weight / (k as f64 + rank as f64);
        entries.insert(
            note_id,
            SearchResult {
                note_id,
                rrf_score,
                semantic_score: Some(score),
                semantic_rank: Some(rank),
                fts_score: None,
                fts_rank: None,
                headline: None,
                rerank_score: None,
                rerank_rank: None,
            },
        );
    }

    for (rank_idx, (note_id, score, headline)) in fts_results.iter().enumerate() {
        let rank = rank_idx + 1; // 1-indexed
        let rrf_contrib = fts_weight / (k as f64 + rank as f64);
        if let Some(entry) = entries.get_mut(note_id) {
            entry.rrf_score += rrf_contrib;
            entry.fts_score = Some(*score);
            entry.fts_rank = Some(rank);
            if headline.is_some() {
                entry.headline.clone_from(headline);
            }
        } else {
            entries.insert(
                *note_id,
                SearchResult {
                    note_id: *note_id,
                    rrf_score: rrf_contrib,
                    semantic_score: None,
                    semantic_rank: None,
                    fts_score: Some(*score),
                    fts_rank: Some(rank),
                    headline: headline.clone(),
                    rerank_score: None,
                    rerank_rank: None,
                },
            );
        }
    }

    let total = entries.len();
    let mut semantic_only = 0;
    let mut fts_only = 0;
    let mut both = 0;
    for entry in entries.values() {
        match (entry.semantic_score.is_some(), entry.fts_score.is_some()) {
            (true, true) => both += 1,
            (true, false) => semantic_only += 1,
            (false, true) => fts_only += 1,
            (false, false) => {}
        }
    }

    let mut results: Vec<SearchResult> = entries.into_values().collect();
    results.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);

    let stats = FusionStats {
        semantic_only,
        fts_only,
        both,
        total,
    };

    (results, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- SearchResult::sources() tests ---

    #[test]
    fn sources_semantic_only() {
        let result = SearchResult {
            note_id: 1,
            rrf_score: 0.5,
            semantic_score: Some(0.9),
            semantic_rank: Some(1),
            fts_score: None,
            fts_rank: None,
            headline: None,
            rerank_score: None,
            rerank_rank: None,
        };
        assert_eq!(result.sources(), vec!["semantic"]);
    }

    #[test]
    fn sources_fts_only() {
        let result = SearchResult {
            note_id: 1,
            rrf_score: 0.5,
            semantic_score: None,
            semantic_rank: None,
            fts_score: Some(0.8),
            fts_rank: Some(1),
            headline: Some("test".to_string()),
            rerank_score: None,
            rerank_rank: None,
        };
        assert_eq!(result.sources(), vec!["fts"]);
    }

    #[test]
    fn sources_both() {
        let result = SearchResult {
            note_id: 1,
            rrf_score: 0.5,
            semantic_score: Some(0.9),
            semantic_rank: Some(1),
            fts_score: Some(0.8),
            fts_rank: Some(2),
            headline: None,
            rerank_score: None,
            rerank_rank: None,
        };
        assert_eq!(result.sources(), vec!["semantic", "fts"]);
    }

    #[test]
    fn sources_neither() {
        let result = SearchResult {
            note_id: 1,
            rrf_score: 0.0,
            semantic_score: None,
            semantic_rank: None,
            fts_score: None,
            fts_rank: None,
            headline: None,
            rerank_score: None,
            rerank_rank: None,
        };
        let sources: Vec<&str> = result.sources();
        assert!(sources.is_empty());
    }

    // --- reciprocal_rank_fusion() tests ---

    #[test]
    fn rrf_empty_inputs_returns_empty() {
        let (results, stats) = reciprocal_rank_fusion(&[], &[], 60, 50, 1.0, 1.0);
        assert!(results.is_empty());
        assert_eq!(
            stats,
            FusionStats {
                semantic_only: 0,
                fts_only: 0,
                both: 0,
                total: 0,
            }
        );
    }

    #[test]
    fn rrf_semantic_only_results() {
        let semantic = vec![(100, 0.95), (200, 0.85), (300, 0.75)];
        let fts: Vec<(i64, f64, Option<String>)> = vec![];

        let (results, stats) = reciprocal_rank_fusion(&semantic, &fts, 60, 50, 1.0, 1.0);

        assert_eq!(results.len(), 3);
        assert_eq!(stats.semantic_only, 3);
        assert_eq!(stats.fts_only, 0);
        assert_eq!(stats.both, 0);
        assert_eq!(stats.total, 3);

        // First result should have highest RRF score (rank 1)
        assert_eq!(results[0].note_id, 100);
        assert!(results[0].semantic_score.is_some());
        assert!(results[0].fts_score.is_none());
    }

    #[test]
    fn rrf_fts_only_results() {
        let semantic: Vec<(i64, f64)> = vec![];
        let fts = vec![(100, 0.9, Some("headline A".to_string())), (200, 0.8, None)];

        let (results, stats) = reciprocal_rank_fusion(&semantic, &fts, 60, 50, 1.0, 1.0);

        assert_eq!(results.len(), 2);
        assert_eq!(stats.fts_only, 2);
        assert_eq!(stats.semantic_only, 0);
        assert_eq!(stats.both, 0);
        assert_eq!(stats.total, 2);

        assert_eq!(results[0].note_id, 100);
        assert_eq!(results[0].headline.as_deref(), Some("headline A"));
    }

    #[test]
    fn rrf_overlapping_results_counts_both() {
        let semantic = vec![(100, 0.95), (200, 0.85)];
        let fts = vec![(200, 0.9, Some("hl".to_string())), (300, 0.8, None)];

        let (results, stats) = reciprocal_rank_fusion(&semantic, &fts, 60, 50, 1.0, 1.0);

        assert_eq!(stats.both, 1); // note_id 200 is in both
        assert_eq!(stats.semantic_only, 1); // note_id 100
        assert_eq!(stats.fts_only, 1); // note_id 300
        assert_eq!(stats.total, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn rrf_score_formula_verified() {
        // With known inputs, verify RRF score = sum(weight / (k + rank))
        let k = 60_usize;
        let semantic = vec![(100, 0.95)]; // rank 1
        let fts = vec![(100, 0.9, None)]; // rank 1

        let (results, _) = reciprocal_rank_fusion(&semantic, &fts, k, 50, 1.0, 1.0);

        assert_eq!(results.len(), 1);
        let expected_rrf = 1.0 / (k as f64 + 1.0) + 1.0 / (k as f64 + 1.0);
        let actual_rrf = results[0].rrf_score;
        assert!(
            (actual_rrf - expected_rrf).abs() < 1e-10,
            "expected {expected_rrf}, got {actual_rrf}"
        );
    }

    #[test]
    fn rrf_score_with_weights() {
        let k = 60_usize;
        let semantic = vec![(100, 0.95)]; // rank 1
        let fts = vec![(100, 0.9, None)]; // rank 1

        let (results, _) = reciprocal_rank_fusion(&semantic, &fts, k, 50, 2.0, 0.5);

        let expected_rrf = 2.0 / (k as f64 + 1.0) + 0.5 / (k as f64 + 1.0);
        let actual_rrf = results[0].rrf_score;
        assert!(
            (actual_rrf - expected_rrf).abs() < 1e-10,
            "expected {expected_rrf}, got {actual_rrf}"
        );
    }

    #[test]
    fn rrf_sorted_descending_by_score() {
        // Note in both sources should rank higher than note in one source
        let semantic = vec![(100, 0.9), (200, 0.8)];
        let fts = vec![(200, 0.7, None), (300, 0.6, None)];

        let (results, _) = reciprocal_rank_fusion(&semantic, &fts, 60, 50, 1.0, 1.0);

        // Verify descending order
        for pair in results.windows(2) {
            assert!(
                pair[0].rrf_score >= pair[1].rrf_score,
                "Results not sorted: {} >= {} failed",
                pair[0].rrf_score,
                pair[1].rrf_score
            );
        }

        // note_id 200 is in both sources, so should have highest score
        assert_eq!(results[0].note_id, 200);
    }

    #[test]
    fn rrf_limit_truncates_results() {
        let semantic: Vec<(i64, f64)> = (1..=10).map(|i| (i, 1.0 - i as f64 * 0.05)).collect();
        let fts: Vec<(i64, f64, Option<String>)> = vec![];

        let (results, stats) = reciprocal_rank_fusion(&semantic, &fts, 60, 3, 1.0, 1.0);

        assert_eq!(results.len(), 3);
        assert_eq!(stats.total, 10); // total includes all unique IDs
    }

    #[test]
    fn rrf_preserves_headline_from_fts() {
        let semantic = vec![(100, 0.9)];
        let fts = vec![(100, 0.8, Some("matched <b>keyword</b>".to_string()))];

        let (results, _) = reciprocal_rank_fusion(&semantic, &fts, 60, 50, 1.0, 1.0);

        assert_eq!(
            results[0].headline.as_deref(),
            Some("matched <b>keyword</b>")
        );
    }

    #[test]
    fn rrf_ranks_are_one_indexed() {
        let semantic = vec![(100, 0.9), (200, 0.8)];
        let fts: Vec<(i64, f64, Option<String>)> = vec![];

        let (results, _) = reciprocal_rank_fusion(&semantic, &fts, 60, 50, 1.0, 1.0);

        // First semantic result should have rank 1
        let first = results.iter().find(|r| r.note_id == 100).unwrap();
        assert_eq!(first.semantic_rank, Some(1));

        let second = results.iter().find(|r| r.note_id == 200).unwrap();
        assert_eq!(second.semantic_rank, Some(2));
    }

    #[test]
    fn rrf_rerank_fields_initially_none() {
        let semantic = vec![(100, 0.9)];
        let (results, _) = reciprocal_rank_fusion(&semantic, &[], 60, 50, 1.0, 1.0);

        assert!(results[0].rerank_score.is_none());
        assert!(results[0].rerank_rank.is_none());
    }
}

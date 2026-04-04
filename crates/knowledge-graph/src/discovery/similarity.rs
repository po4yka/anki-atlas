use tracing::instrument;

use crate::error::KnowledgeGraphError;
use crate::models::{ConceptEdge, EdgeSource, EdgeType};
use crate::repository::KnowledgeGraphRepository;

/// Minimum cosine similarity for creating a "similar" edge.
const MIN_SIMILARITY: f32 = 0.7;

/// Maximum similarity before it's considered a duplicate (handled elsewhere).
const MAX_SIMILARITY: f32 = 0.93;

/// Maximum similar notes to find per source note.
const SIMILAR_LIMIT: usize = 10;

/// Discover similarity edges between notes using vector embeddings.
///
/// For each note, finds the top-K most similar notes via Qdrant and creates
/// bidirectional `Similar` edges. Notes above the duplicate threshold (0.93)
/// are excluded since those are handled by the duplicate detector.
///
/// This function accepts pre-computed similarity pairs to decouple from Qdrant.
/// The caller is responsible for querying `VectorRepository::find_similar_to_note`.
#[instrument(skip_all, fields(pairs = similarity_pairs.len()))]
pub async fn discover_similarity_edges(
    repo: &dyn KnowledgeGraphRepository,
    similarity_pairs: &[(i64, i64, f32)], // (source_note_id, target_note_id, score)
) -> Result<usize, KnowledgeGraphError> {
    let mut edges = Vec::new();

    for &(source_id, target_id, score) in similarity_pairs {
        if !(MIN_SIMILARITY..=MAX_SIMILARITY).contains(&score) {
            continue;
        }
        if source_id == target_id {
            continue;
        }

        // Bidirectional edges
        edges.push(ConceptEdge {
            source_note_id: source_id,
            target_note_id: target_id,
            edge_type: EdgeType::Similar,
            edge_source: EdgeSource::Embedding,
            weight: score,
        });
        edges.push(ConceptEdge {
            source_note_id: target_id,
            target_note_id: source_id,
            edge_type: EdgeType::Similar,
            edge_source: EdgeSource::Embedding,
            weight: score,
        });
    }

    if edges.is_empty() {
        return Ok(0);
    }

    repo.upsert_concept_edges(&edges).await
}

/// Configuration for similarity discovery.
pub struct SimilarityConfig {
    pub min_similarity: f32,
    pub max_similarity: f32,
    pub limit_per_note: usize,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            min_similarity: MIN_SIMILARITY,
            max_similarity: MAX_SIMILARITY,
            limit_per_note: SIMILAR_LIMIT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repository::MockKnowledgeGraphRepository;

    #[tokio::test]
    async fn filters_below_min_similarity() {
        let mut mock = MockKnowledgeGraphRepository::new();
        mock.expect_upsert_concept_edges().never();

        let pairs = vec![(1, 2, 0.5)]; // below MIN_SIMILARITY
        let count = discover_similarity_edges(&mock, &pairs).await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn filters_above_max_similarity() {
        let mut mock = MockKnowledgeGraphRepository::new();
        mock.expect_upsert_concept_edges().never();

        let pairs = vec![(1, 2, 0.95)]; // above MAX_SIMILARITY (duplicate territory)
        let count = discover_similarity_edges(&mock, &pairs).await.unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn creates_bidirectional_edges() {
        let mut mock = MockKnowledgeGraphRepository::new();
        mock.expect_upsert_concept_edges()
            .withf(|edges| edges.len() == 2) // bidirectional
            .returning(|edges| {
                let len = edges.len();
                Box::pin(async move { Ok(len) })
            });

        let pairs = vec![(1, 2, 0.85)];
        let count = discover_similarity_edges(&mock, &pairs).await.unwrap();
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn skips_self_edges() {
        let mut mock = MockKnowledgeGraphRepository::new();
        mock.expect_upsert_concept_edges().never();

        let pairs = vec![(1, 1, 0.99)]; // same note
        let count = discover_similarity_edges(&mock, &pairs).await.unwrap();
        assert_eq!(count, 0);
    }
}

use std::collections::HashSet;

use crate::error::KnowledgeGraphError;
use crate::models::{ConceptEdge, TopicEdge};
use crate::repository::KnowledgeGraphRepository;

/// Find "see also" recommendations for a note.
///
/// Returns related notes sorted by weight, excluding duplicates.
pub async fn see_also(
    repo: &dyn KnowledgeGraphRepository,
    note_id: i64,
    limit: usize,
) -> Result<Vec<ConceptEdge>, KnowledgeGraphError> {
    repo.get_related_notes(note_id, None, limit).await
}

/// Find all prerequisite notes for a given note (direct prerequisites only).
pub async fn prerequisites(
    repo: &dyn KnowledgeGraphRepository,
    note_id: i64,
) -> Result<Vec<ConceptEdge>, KnowledgeGraphError> {
    repo.get_prerequisites(note_id).await
}

/// Recursively find the prerequisite chain up to `max_depth` levels deep.
///
/// Returns a flat list of all prerequisite edges discovered via BFS.
pub async fn prerequisite_chain(
    repo: &dyn KnowledgeGraphRepository,
    note_id: i64,
    max_depth: usize,
) -> Result<Vec<ConceptEdge>, KnowledgeGraphError> {
    let mut all_edges = Vec::new();
    let mut visited = HashSet::new();
    let mut frontier = vec![note_id];
    visited.insert(note_id);

    for _depth in 0..max_depth {
        if frontier.is_empty() {
            break;
        }

        let mut next_frontier = Vec::new();
        for current_id in &frontier {
            let prereqs = repo.get_prerequisites(*current_id).await?;
            for edge in prereqs {
                if visited.insert(edge.source_note_id) {
                    next_frontier.push(edge.source_note_id);
                }
                all_edges.push(edge);
            }
        }
        frontier = next_frontier;
    }

    Ok(all_edges)
}

/// Get the neighborhood subgraph for a topic (all connected topics within N hops).
///
/// Returns the edges and the set of topic IDs in the neighborhood.
pub async fn topic_neighborhood(
    repo: &dyn KnowledgeGraphRepository,
    topic_id: i32,
    max_hops: usize,
    limit_per_hop: usize,
) -> Result<(Vec<TopicEdge>, Vec<i32>), KnowledgeGraphError> {
    let mut all_edges = Vec::new();
    let mut visited = HashSet::new();
    let mut frontier = vec![topic_id];
    visited.insert(topic_id);

    for _hop in 0..max_hops {
        if frontier.is_empty() {
            break;
        }

        let mut next_frontier = Vec::new();
        for &current_id in &frontier {
            let edges = repo
                .get_related_topics(current_id, None, limit_per_hop)
                .await?;
            for edge in edges {
                let neighbor = if edge.source_topic_id == current_id {
                    edge.target_topic_id
                } else {
                    edge.source_topic_id
                };
                if visited.insert(neighbor) {
                    next_frontier.push(neighbor);
                }
                all_edges.push(edge);
            }
        }
        frontier = next_frontier;
    }

    let topic_ids: Vec<i32> = visited.into_iter().collect();
    Ok((all_edges, topic_ids))
}

/// Detect prerequisite gaps: topics that are prerequisites but have low coverage.
pub async fn prerequisite_gaps(
    repo: &dyn KnowledgeGraphRepository,
    topic_id: i32,
) -> Result<Vec<TopicEdge>, KnowledgeGraphError> {
    repo.get_topic_prerequisites(topic_id).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{EdgeSource, EdgeType};
    use crate::repository::MockKnowledgeGraphRepository;

    fn make_prereq_edge(source: i64, target: i64, weight: f32) -> ConceptEdge {
        ConceptEdge {
            source_note_id: source,
            target_note_id: target,
            edge_type: EdgeType::Prerequisite,
            edge_source: EdgeSource::ReviewInference,
            weight,
        }
    }

    #[tokio::test]
    async fn prerequisite_chain_traverses_bfs() {
        let mut mock = MockKnowledgeGraphRepository::new();

        mock.expect_get_prerequisites()
            .withf(|id| *id == 3)
            .returning(|_| Box::pin(async { Ok(vec![make_prereq_edge(2, 3, 0.8)]) }));
        mock.expect_get_prerequisites()
            .withf(|id| *id == 2)
            .returning(|_| Box::pin(async { Ok(vec![make_prereq_edge(1, 2, 0.9)]) }));
        mock.expect_get_prerequisites()
            .withf(|id| *id == 1)
            .returning(|_| Box::pin(async { Ok(vec![]) }));

        let chain = prerequisite_chain(&mock, 3, 5).await.unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].source_note_id, 2);
        assert_eq!(chain[1].source_note_id, 1);
    }

    #[tokio::test]
    async fn prerequisite_chain_respects_max_depth() {
        let mut mock = MockKnowledgeGraphRepository::new();

        mock.expect_get_prerequisites()
            .withf(|id| *id == 3)
            .returning(|_| Box::pin(async { Ok(vec![make_prereq_edge(2, 3, 0.8)]) }));

        let chain = prerequisite_chain(&mock, 3, 1).await.unwrap();
        assert_eq!(chain.len(), 1);
    }
}

use serde::Serialize;

use crate::AnalyticsError;

/// A single duplicate note within a cluster.
#[derive(Debug, Clone, Serialize)]
pub struct DuplicateDetail {
    pub note_id: i64,
    pub similarity: f64,
    pub text: String,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
}

/// A cluster of near-duplicate notes.
#[derive(Debug, Clone, Serialize)]
pub struct DuplicateCluster {
    pub representative_id: i64,
    pub representative_text: String,
    pub duplicates: Vec<DuplicateDetail>,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
}

impl DuplicateCluster {
    /// Total notes: representative + duplicates.
    pub fn size(&self) -> usize {
        todo!()
    }
}

/// Statistics from duplicate detection.
#[derive(Debug, Clone, Default, Serialize)]
pub struct DuplicateStats {
    pub notes_scanned: usize,
    pub clusters_found: usize,
    pub total_duplicates: usize,
    pub avg_cluster_size: f64,
}

/// Duplicate detector. Uses VectorRepository for similarity search.
pub struct DuplicateDetector<V: indexer::qdrant::VectorRepository> {
    pub vector_repo: V,
    pub db: sqlx::PgPool,
}

impl<V: indexer::qdrant::VectorRepository> DuplicateDetector<V> {
    pub fn new(vector_repo: V, db: sqlx::PgPool) -> Self {
        todo!()
    }

    /// Find clusters of near-duplicate notes.
    pub async fn find_duplicates(
        &self,
        _threshold: f64,
        _max_clusters: usize,
        _deck_filter: Option<&[String]>,
        _tag_filter: Option<&[String]>,
    ) -> Result<(Vec<DuplicateCluster>, DuplicateStats), AnalyticsError> {
        todo!()
    }
}

/// Internal: union-find with path compression.
/// Always uses smaller ID as root for determinism.
pub(crate) struct UnionFind {
    parent: std::collections::HashMap<i64, i64>,
}

impl UnionFind {
    pub(crate) fn new() -> Self {
        todo!()
    }

    pub(crate) fn find(&mut self, _x: i64) -> i64 {
        todo!()
    }

    pub(crate) fn union(&mut self, _x: i64, _y: i64) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- DuplicateCluster::size ---

    #[test]
    fn cluster_size_no_duplicates() {
        let cluster = DuplicateCluster {
            representative_id: 1,
            representative_text: "test".to_string(),
            duplicates: vec![],
            deck_names: vec![],
            tags: vec![],
        };
        assert_eq!(cluster.size(), 1);
    }

    #[test]
    fn cluster_size_with_duplicates() {
        let cluster = DuplicateCluster {
            representative_id: 1,
            representative_text: "test".to_string(),
            duplicates: vec![
                DuplicateDetail {
                    note_id: 2,
                    similarity: 0.95,
                    text: "dup1".to_string(),
                    deck_names: vec![],
                    tags: vec![],
                },
                DuplicateDetail {
                    note_id: 3,
                    similarity: 0.93,
                    text: "dup2".to_string(),
                    deck_names: vec![],
                    tags: vec![],
                },
            ],
            deck_names: vec![],
            tags: vec![],
        };
        assert_eq!(cluster.size(), 3);
    }

    // --- UnionFind ---

    #[test]
    fn union_find_single_element() {
        let mut uf = UnionFind::new();
        assert_eq!(uf.find(5), 5);
    }

    #[test]
    fn union_find_uses_smaller_id_as_root() {
        let mut uf = UnionFind::new();
        uf.union(10, 5);
        assert_eq!(uf.find(10), 5);
        assert_eq!(uf.find(5), 5);
    }

    #[test]
    fn union_find_uses_smaller_id_as_root_reverse_order() {
        let mut uf = UnionFind::new();
        uf.union(3, 7);
        assert_eq!(uf.find(7), 3);
        assert_eq!(uf.find(3), 3);
    }

    #[test]
    fn union_find_transitive() {
        let mut uf = UnionFind::new();
        uf.union(1, 2);
        uf.union(2, 3);
        // All should resolve to 1 (smallest)
        assert_eq!(uf.find(1), 1);
        assert_eq!(uf.find(2), 1);
        assert_eq!(uf.find(3), 1);
    }

    #[test]
    fn union_find_merge_two_groups() {
        let mut uf = UnionFind::new();
        // Group 1: {5, 10}
        uf.union(5, 10);
        // Group 2: {3, 7}
        uf.union(3, 7);
        // Merge: smaller root wins => 3
        uf.union(5, 7);
        assert_eq!(uf.find(5), 3);
        assert_eq!(uf.find(10), 3);
        assert_eq!(uf.find(3), 3);
        assert_eq!(uf.find(7), 3);
    }

    #[test]
    fn union_find_idempotent() {
        let mut uf = UnionFind::new();
        uf.union(1, 2);
        uf.union(1, 2);
        assert_eq!(uf.find(1), 1);
        assert_eq!(uf.find(2), 1);
    }
}

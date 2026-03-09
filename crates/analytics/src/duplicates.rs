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
        1 + self.duplicates.len()
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
        Self { vector_repo, db }
    }

    /// Find clusters of near-duplicate notes.
    pub async fn find_duplicates(
        &self,
        threshold: f64,
        max_clusters: usize,
        deck_filter: Option<&[String]>,
        tag_filter: Option<&[String]>,
    ) -> Result<(Vec<DuplicateCluster>, DuplicateStats), AnalyticsError> {
        // Fetch all active note IDs
        let note_ids: Vec<(i64,)> = sqlx::query_as(
            "SELECT note_id FROM notes WHERE deleted_at IS NULL ORDER BY note_id",
        )
        .fetch_all(&self.db)
        .await?;

        let mut uf = UnionFind::new();
        // Track similarity pairs: (note_a, note_b) -> score
        let mut pair_scores: std::collections::HashMap<(i64, i64), f64> =
            std::collections::HashMap::new();

        for (note_id,) in &note_ids {
            let similar = self
                .vector_repo
                .find_similar_to_note(
                    *note_id,
                    20,
                    threshold as f32,
                    deck_filter,
                    tag_filter,
                )
                .await?;

            for (other_id, score) in similar {
                if other_id == *note_id {
                    continue;
                }
                let pair = if *note_id < other_id {
                    (*note_id, other_id)
                } else {
                    (other_id, *note_id)
                };
                pair_scores
                    .entry(pair)
                    .or_insert(f64::from(score));
                uf.union(*note_id, other_id);
            }
        }

        // Group notes by cluster root
        let mut cluster_members: std::collections::HashMap<i64, Vec<i64>> =
            std::collections::HashMap::new();
        for (note_id,) in &note_ids {
            let root = uf.find(*note_id);
            cluster_members.entry(root).or_default().push(*note_id);
        }

        // Build clusters (only groups with 2+ members)
        let mut clusters = Vec::new();
        for members in cluster_members.values() {
            if members.len() < 2 {
                continue;
            }

            // Pick representative: note with most reviews
            let mut best_id = members[0];
            let mut best_reviews: i64 = 0;

            for &nid in members {
                let reviews: (i64,) = sqlx::query_as(
                    "SELECT COALESCE(SUM(c.reps), 0) FROM cards c WHERE c.note_id = $1",
                )
                .bind(nid)
                .fetch_one(&self.db)
                .await?;
                if reviews.0 > best_reviews {
                    best_reviews = reviews.0;
                    best_id = nid;
                }
            }

            // Fetch note details
            let (rep_text,): (String,) = sqlx::query_as(
                "SELECT LEFT(normalized_text, 200) FROM notes WHERE note_id = $1",
            )
            .bind(best_id)
            .fetch_one(&self.db)
            .await?;

            let mut duplicates = Vec::new();
            let mut all_decks = Vec::new();
            let mut all_tags = Vec::new();

            for &nid in members {
                if nid == best_id {
                    continue;
                }

                let pair = if best_id < nid {
                    (best_id, nid)
                } else {
                    (nid, best_id)
                };
                let sim = pair_scores.get(&pair).copied().unwrap_or(0.0);

                let detail_row: (String, Vec<String>) = sqlx::query_as(
                    "SELECT LEFT(n.normalized_text, 200), COALESCE(n.tags, '{}') \
                     FROM notes n WHERE n.note_id = $1",
                )
                .bind(nid)
                .fetch_one(&self.db)
                .await?;

                let deck_names: Vec<(String,)> = sqlx::query_as(
                    "SELECT DISTINCT d.name FROM cards c \
                     JOIN decks d ON d.deck_id = c.deck_id \
                     WHERE c.note_id = $1",
                )
                .bind(nid)
                .fetch_all(&self.db)
                .await?;

                let dn: Vec<String> = deck_names.into_iter().map(|(n,)| n).collect();
                all_decks.extend(dn.clone());
                all_tags.extend(detail_row.1.clone());

                duplicates.push(DuplicateDetail {
                    note_id: nid,
                    similarity: sim,
                    text: detail_row.0,
                    deck_names: dn,
                    tags: detail_row.1,
                });
            }

            // Get representative's decks and tags
            let rep_decks: Vec<(String,)> = sqlx::query_as(
                "SELECT DISTINCT d.name FROM cards c \
                 JOIN decks d ON d.deck_id = c.deck_id \
                 WHERE c.note_id = $1",
            )
            .bind(best_id)
            .fetch_all(&self.db)
            .await?;
            let rep_tags: (Vec<String>,) = sqlx::query_as(
                "SELECT COALESCE(tags, '{}') FROM notes WHERE note_id = $1",
            )
            .bind(best_id)
            .fetch_one(&self.db)
            .await?;

            all_decks.extend(rep_decks.iter().map(|(n,)| n.clone()));
            all_tags.extend(rep_tags.0.clone());
            all_decks.sort();
            all_decks.dedup();
            all_tags.sort();
            all_tags.dedup();

            clusters.push(DuplicateCluster {
                representative_id: best_id,
                representative_text: rep_text,
                duplicates,
                deck_names: all_decks,
                tags: all_tags,
            });
        }

        clusters.sort_by_key(|b| std::cmp::Reverse(b.size()));
        clusters.truncate(max_clusters);

        let total_duplicates: usize = clusters.iter().map(|c| c.duplicates.len()).sum();
        let avg_cluster_size = if clusters.is_empty() {
            0.0
        } else {
            clusters.iter().map(|c| c.size()).sum::<usize>() as f64 / clusters.len() as f64
        };

        let stats = DuplicateStats {
            notes_scanned: note_ids.len(),
            clusters_found: clusters.len(),
            total_duplicates,
            avg_cluster_size,
        };

        Ok((clusters, stats))
    }
}

/// Internal: union-find with path compression.
/// Always uses smaller ID as root for determinism.
/// Used by `DuplicateDetector::find_duplicates` (implementation pending).
#[allow(dead_code)]
pub(crate) struct UnionFind {
    parent: std::collections::HashMap<i64, i64>,
}

#[allow(dead_code)]
impl UnionFind {
    pub(crate) fn new() -> Self {
        Self {
            parent: std::collections::HashMap::new(),
        }
    }

    pub(crate) fn find(&mut self, x: i64) -> i64 {
        match self.parent.entry(x) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(x);
                return x;
            }
            std::collections::hash_map::Entry::Occupied(e) => {
                let p = *e.get();
                if p == x {
                    return x;
                }
            }
        }
        let p = self.parent[&x];
        let root = self.find(p);
        self.parent.insert(x, root);
        root
    }

    pub(crate) fn union(&mut self, x: i64, y: i64) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        // Smaller ID becomes root for determinism.
        let (new_root, child) = if rx < ry { (rx, ry) } else { (ry, rx) };
        self.parent.insert(child, new_root);
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

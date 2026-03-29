use std::collections::HashMap;
use std::sync::Arc;

use serde::Serialize;

use crate::AnalyticsError;
use crate::repository::AnalyticsRepository;

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
    pub repository: Arc<dyn AnalyticsRepository>,
}

impl<V: indexer::qdrant::VectorRepository> DuplicateDetector<V> {
    pub fn new(vector_repo: V, repository: Arc<dyn AnalyticsRepository>) -> Self {
        Self {
            vector_repo,
            repository,
        }
    }

    async fn fetch_active_note_ids(&self) -> Result<Vec<i64>, AnalyticsError> {
        self.repository.fetch_active_note_ids().await
    }

    async fn collect_similarity_pairs(
        &self,
        note_ids: &[i64],
        threshold: f64,
        deck_filter: Option<&[String]>,
        tag_filter: Option<&[String]>,
    ) -> Result<(UnionFind, HashMap<(i64, i64), f64>), AnalyticsError> {
        let mut union_find = UnionFind::new();
        let mut pair_scores = HashMap::new();
        let mut first_error = None;
        let mut successful_lookups = 0_usize;

        for &note_id in note_ids {
            let similar_notes = match self
                .vector_repo
                .find_similar_to_note(note_id, 20, threshold as f32, deck_filter, tag_filter)
                .await
            {
                Ok(similar_notes) => {
                    successful_lookups += 1;
                    similar_notes
                }
                Err(error) => {
                    tracing::warn!(
                        note_id,
                        error = %error,
                        "skipping duplicate similarity lookup for note"
                    );
                    if first_error.is_none() {
                        first_error = Some(AnalyticsError::from(error));
                    }
                    continue;
                }
            };

            for (other_note_id, score) in similar_notes {
                if other_note_id == note_id {
                    continue;
                }

                let note_pair = if note_id < other_note_id {
                    (note_id, other_note_id)
                } else {
                    (other_note_id, note_id)
                };

                pair_scores.entry(note_pair).or_insert(f64::from(score));
                union_find.union(note_id, other_note_id);
            }
        }

        if successful_lookups == 0 && !note_ids.is_empty() {
            return Err(first_error.unwrap_or_else(|| {
                AnalyticsError::VectorStore(indexer::qdrant::VectorStoreError::Client(
                    "duplicate similarity lookup failed for all notes".to_string(),
                ))
            }));
        }

        Ok((union_find, pair_scores))
    }

    fn group_cluster_members(
        note_ids: &[i64],
        union_find: &mut UnionFind,
    ) -> HashMap<i64, Vec<i64>> {
        let mut cluster_members: HashMap<i64, Vec<i64>> = HashMap::new();
        for &note_id in note_ids {
            let cluster_root = union_find.find(note_id);
            cluster_members
                .entry(cluster_root)
                .or_default()
                .push(note_id);
        }
        cluster_members
    }

    async fn select_representative_id(
        &self,
        member_note_ids: &[i64],
    ) -> Result<i64, AnalyticsError> {
        let mut representative_id = member_note_ids[0];
        let mut highest_review_count = 0_i64;

        for &note_id in member_note_ids {
            let review_count = self.repository.fetch_note_review_count(note_id).await?;

            if review_count > highest_review_count {
                highest_review_count = review_count;
                representative_id = note_id;
            }
        }

        Ok(representative_id)
    }

    async fn fetch_note_excerpt_and_tags(
        &self,
        note_id: i64,
    ) -> Result<(String, Vec<String>), AnalyticsError> {
        self.repository.fetch_note_excerpt_and_tags(note_id).await
    }

    async fn fetch_note_deck_names(&self, note_id: i64) -> Result<Vec<String>, AnalyticsError> {
        self.repository.fetch_note_deck_names(note_id).await
    }

    async fn build_cluster(
        &self,
        member_note_ids: &[i64],
        pair_scores: &HashMap<(i64, i64), f64>,
    ) -> Result<Option<DuplicateCluster>, AnalyticsError> {
        if member_note_ids.len() < 2 {
            return Ok(None);
        }

        let representative_id = self.select_representative_id(member_note_ids).await?;
        let (representative_text, representative_tags) =
            self.fetch_note_excerpt_and_tags(representative_id).await?;
        let representative_deck_names = self.fetch_note_deck_names(representative_id).await?;

        let mut duplicate_details = Vec::new();
        let mut cluster_deck_names = representative_deck_names.clone();
        let mut cluster_tags = representative_tags.clone();

        for &note_id in member_note_ids {
            if note_id == representative_id {
                continue;
            }

            let note_pair = if representative_id < note_id {
                (representative_id, note_id)
            } else {
                (note_id, representative_id)
            };
            let similarity = pair_scores.get(&note_pair).copied().unwrap_or(0.0);
            let (duplicate_text, duplicate_tags) =
                self.fetch_note_excerpt_and_tags(note_id).await?;
            let duplicate_deck_names = self.fetch_note_deck_names(note_id).await?;

            cluster_deck_names.extend(duplicate_deck_names.clone());
            cluster_tags.extend(duplicate_tags.clone());
            duplicate_details.push(DuplicateDetail {
                note_id,
                similarity,
                text: duplicate_text,
                deck_names: duplicate_deck_names,
                tags: duplicate_tags,
            });
        }

        cluster_deck_names.sort();
        cluster_deck_names.dedup();
        cluster_tags.sort();
        cluster_tags.dedup();

        Ok(Some(DuplicateCluster {
            representative_id,
            representative_text,
            duplicates: duplicate_details,
            deck_names: cluster_deck_names,
            tags: cluster_tags,
        }))
    }

    fn summarize_clusters(
        note_ids_scanned: usize,
        clusters: &[DuplicateCluster],
    ) -> DuplicateStats {
        let total_duplicates: usize = clusters
            .iter()
            .map(|cluster| cluster.duplicates.len())
            .sum();
        let avg_cluster_size = if clusters.is_empty() {
            0.0
        } else {
            clusters.iter().map(DuplicateCluster::size).sum::<usize>() as f64
                / clusters.len() as f64
        };

        DuplicateStats {
            notes_scanned: note_ids_scanned,
            clusters_found: clusters.len(),
            total_duplicates,
            avg_cluster_size,
        }
    }

    /// Find clusters of near-duplicate notes.
    pub async fn find_duplicates(
        &self,
        threshold: f64,
        max_clusters: usize,
        deck_filter: Option<&[String]>,
        tag_filter: Option<&[String]>,
    ) -> Result<(Vec<DuplicateCluster>, DuplicateStats), AnalyticsError> {
        let note_ids = self.fetch_active_note_ids().await?;
        let (mut union_find, pair_scores) = self
            .collect_similarity_pairs(&note_ids, threshold, deck_filter, tag_filter)
            .await?;
        let cluster_members = Self::group_cluster_members(&note_ids, &mut union_find);

        let mut clusters = Vec::new();
        for member_note_ids in cluster_members.values() {
            if let Some(cluster) = self.build_cluster(member_note_ids, &pair_scores).await? {
                clusters.push(cluster);
            }
        }

        clusters.sort_by_key(|cluster| std::cmp::Reverse(cluster.size()));
        clusters.truncate(max_clusters);

        let stats = Self::summarize_clusters(note_ids.len(), &clusters);

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
    use async_trait::async_trait;
    use indexer::qdrant::{
        NotePayload, SearchFilters, SemanticSearchHit, SparseVector, VectorRepository,
        VectorStoreError,
    };
    use crate::repository::SqlxAnalyticsRepository;
    use sqlx::postgres::PgPoolOptions;

    #[derive(Debug)]
    struct StubVectorRepo {
        results: HashMap<i64, StubResult>,
    }

    #[derive(Debug, Clone)]
    enum StubResult {
        Ok(Vec<(i64, f32)>),
        ClientError(&'static str),
        ConnectionError(&'static str),
    }

    #[async_trait]
    impl VectorRepository for StubVectorRepo {
        async fn ensure_collection(&self, _dimension: usize) -> Result<bool, VectorStoreError> {
            Ok(false)
        }

        async fn collection_dimension(&self) -> Result<Option<usize>, VectorStoreError> {
            Ok(Some(384))
        }

        async fn recreate_collection(&self, _dimension: usize) -> Result<(), VectorStoreError> {
            Ok(())
        }

        async fn upsert_vectors(
            &self,
            _vectors: &[Vec<f32>],
            _payloads: &[NotePayload],
            _sparse_vectors: Option<&[SparseVector]>,
        ) -> Result<usize, VectorStoreError> {
            Ok(0)
        }

        async fn delete_vectors(&self, _note_ids: &[i64]) -> Result<usize, VectorStoreError> {
            Ok(0)
        }

        async fn get_existing_hashes(
            &self,
            _note_ids: &[i64],
        ) -> Result<HashMap<i64, String>, VectorStoreError> {
            Ok(HashMap::new())
        }

        async fn search_chunks(
            &self,
            _query_vector: &[f32],
            _query_sparse: Option<&SparseVector>,
            _limit: usize,
            _filters: &SearchFilters,
        ) -> Result<Vec<SemanticSearchHit>, VectorStoreError> {
            Ok(Vec::new())
        }

        async fn find_similar_to_note(
            &self,
            note_id: i64,
            _limit: usize,
            _min_score: f32,
            _deck_names: Option<&[String]>,
            _tags: Option<&[String]>,
        ) -> Result<Vec<(i64, f32)>, VectorStoreError> {
            match self.results.get(&note_id).cloned() {
                Some(StubResult::Ok(results)) => Ok(results),
                Some(StubResult::ClientError(message)) => {
                    Err(VectorStoreError::Client(message.to_string()))
                }
                Some(StubResult::ConnectionError(message)) => {
                    Err(VectorStoreError::Connection(message.to_string()))
                }
                None => Ok(Vec::new()),
            }
        }

        async fn close(&self) -> Result<(), VectorStoreError> {
            Ok(())
        }
    }

    fn test_pool() -> sqlx::PgPool {
        PgPoolOptions::new()
            .connect_lazy("postgresql://postgres:postgres@localhost/postgres")
            .expect("create lazy pg pool")
    }

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

    #[tokio::test]
    async fn collect_similarity_pairs_skips_partial_vector_errors() {
        let detector = DuplicateDetector::new(
            StubVectorRepo {
                results: HashMap::from([
                    (1, StubResult::Ok(vec![(2, 0.995)])),
                    (2, StubResult::ClientError("transient qdrant error")),
                ]),
            },
            Arc::new(SqlxAnalyticsRepository::new(test_pool())),
        );

        let (mut union_find, pair_scores) = detector
            .collect_similarity_pairs(&[1, 2], 0.99, None, None)
            .await
            .expect("partial failures should be tolerated");

        let similarity = pair_scores
            .get(&(1, 2))
            .copied()
            .expect("expected similarity pair");
        assert!((similarity - 0.995).abs() < 1e-6);
        assert_eq!(union_find.find(1), union_find.find(2));
    }

    #[tokio::test]
    async fn collect_similarity_pairs_propagates_when_all_vector_lookups_fail() {
        let detector = DuplicateDetector::new(
            StubVectorRepo {
                results: HashMap::from([
                    (1, StubResult::ClientError("qdrant unavailable")),
                    (2, StubResult::ConnectionError("connection reset")),
                ]),
            },
            Arc::new(SqlxAnalyticsRepository::new(test_pool())),
        );

        let error = match detector
            .collect_similarity_pairs(&[1, 2], 0.99, None, None)
            .await
        {
            Ok(_) => panic!("all failures should still propagate"),
            Err(error) => error,
        };

        assert!(
            error.to_string().contains("qdrant unavailable"),
            "unexpected error: {error}"
        );
    }
}

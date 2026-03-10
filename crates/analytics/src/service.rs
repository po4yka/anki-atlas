use crate::AnalyticsError;
use crate::coverage::{TopicCoverage, TopicGap, WeakNote};
use crate::duplicates::{DuplicateCluster, DuplicateDetector, DuplicateStats};
use crate::labeling::{LabelingStats, TopicLabeler};
use crate::taxonomy::Taxonomy;

/// Facade aggregating taxonomy, coverage, labeling, and duplicate detection.
pub struct AnalyticsService<E, V>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
{
    pub embedding: E,
    pub vector_repo: V,
    pub db: sqlx::PgPool,
}

impl<E, V> AnalyticsService<E, V>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
{
    pub fn new(embedding: E, vector_repo: V, db: sqlx::PgPool) -> Self {
        Self {
            embedding,
            vector_repo,
            db,
        }
    }

    /// Load taxonomy from YAML (syncing to DB) or from DB.
    pub async fn load_taxonomy(
        &self,
        yaml_path: Option<&std::path::Path>,
    ) -> Result<Taxonomy, AnalyticsError> {
        crate::taxonomy::load_taxonomy(&self.db, yaml_path).await
    }

    /// Label all notes with topics.
    pub async fn label_notes(
        &self,
        taxonomy: Option<&Taxonomy>,
        min_confidence: f32,
    ) -> Result<LabelingStats, AnalyticsError> {
        let owned_taxonomy;
        let tax = match taxonomy {
            Some(t) => t,
            None => {
                owned_taxonomy = crate::taxonomy::load_taxonomy_from_db(&self.db).await?;
                &owned_taxonomy
            }
        };

        TopicLabeler::new(&self.embedding, self.db.clone())
            .label_notes(tax, min_confidence, 5, 100)
            .await
    }

    /// Get coverage metrics for a topic.
    pub async fn get_coverage(
        &self,
        topic_path: &str,
        include_subtree: bool,
    ) -> Result<Option<TopicCoverage>, AnalyticsError> {
        crate::coverage::get_topic_coverage(&self.db, topic_path, include_subtree).await
    }

    /// Find gaps in topic coverage under a root path.
    pub async fn get_gaps(
        &self,
        topic_path: &str,
        min_coverage: i64,
    ) -> Result<Vec<TopicGap>, AnalyticsError> {
        crate::coverage::get_topic_gaps(&self.db, topic_path, min_coverage).await
    }

    /// Get weak notes (high lapse rate) in a topic subtree.
    pub async fn get_weak_notes(
        &self,
        topic_path: &str,
        max_results: i64,
    ) -> Result<Vec<WeakNote>, AnalyticsError> {
        crate::coverage::get_weak_notes(&self.db, topic_path, max_results, 0.1).await
    }

    /// Find clusters of near-duplicate notes.
    pub async fn find_duplicates(
        &self,
        threshold: f64,
        max_clusters: usize,
        deck_filter: Option<&[String]>,
        tag_filter: Option<&[String]>,
    ) -> Result<(Vec<DuplicateCluster>, DuplicateStats), AnalyticsError> {
        DuplicateDetector::new(&self.vector_repo, self.db.clone())
            .find_duplicates(threshold, max_clusters, deck_filter, tag_filter)
            .await
    }

    /// Get coverage tree for all topics (optionally filtered by root path).
    pub async fn get_taxonomy_tree(
        &self,
        root_path: Option<&str>,
    ) -> Result<Vec<serde_json::Value>, AnalyticsError> {
        crate::coverage::get_coverage_tree(&self.db, root_path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify all public types are Send + Sync.
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn analytics_error_is_send_sync() {
        assert_send_sync::<AnalyticsError>();
    }

    #[test]
    fn topic_is_send_sync() {
        assert_send_sync::<crate::taxonomy::Topic>();
    }

    #[test]
    fn taxonomy_is_send_sync() {
        assert_send_sync::<crate::taxonomy::Taxonomy>();
    }

    #[test]
    fn topic_coverage_is_send_sync() {
        assert_send_sync::<TopicCoverage>();
    }

    #[test]
    fn topic_gap_is_send_sync() {
        assert_send_sync::<TopicGap>();
    }

    #[test]
    fn weak_note_is_send_sync() {
        assert_send_sync::<WeakNote>();
    }

    #[test]
    fn labeling_stats_is_send_sync() {
        assert_send_sync::<LabelingStats>();
    }

    #[test]
    fn duplicate_cluster_is_send_sync() {
        assert_send_sync::<DuplicateCluster>();
    }

    #[test]
    fn duplicate_stats_is_send_sync() {
        assert_send_sync::<DuplicateStats>();
    }

    #[test]
    fn analytics_service_is_generic() {
        // Verify the service compiles with mock types
        // (this is a compile-time check, the test body doesn't matter)
        fn _check<
            E: indexer::embeddings::EmbeddingProvider,
            V: indexer::qdrant::VectorRepository,
        >() {
            assert_send_sync::<AnalyticsService<E, V>>();
        }
    }
}

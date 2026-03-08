use crate::coverage::{TopicCoverage, TopicGap, WeakNote};
use crate::duplicates::{DuplicateCluster, DuplicateStats};
use crate::labeling::LabelingStats;
use crate::taxonomy::Taxonomy;
use crate::AnalyticsError;

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
        _yaml_path: Option<&std::path::Path>,
    ) -> Result<Taxonomy, AnalyticsError> {
        todo!()
    }

    /// Label all notes with topics.
    pub async fn label_notes(
        &self,
        _taxonomy: Option<&Taxonomy>,
        _min_confidence: f32,
    ) -> Result<LabelingStats, AnalyticsError> {
        todo!()
    }

    pub async fn get_coverage(
        &self,
        _topic_path: &str,
        _include_subtree: bool,
    ) -> Result<Option<TopicCoverage>, AnalyticsError> {
        todo!()
    }

    pub async fn get_gaps(
        &self,
        _topic_path: &str,
        _min_coverage: i64,
    ) -> Result<Vec<TopicGap>, AnalyticsError> {
        todo!()
    }

    pub async fn get_weak_notes(
        &self,
        _topic_path: &str,
        _max_results: i64,
    ) -> Result<Vec<WeakNote>, AnalyticsError> {
        todo!()
    }

    pub async fn find_duplicates(
        &self,
        _threshold: f64,
        _max_clusters: usize,
        _deck_filter: Option<&[String]>,
        _tag_filter: Option<&[String]>,
    ) -> Result<(Vec<DuplicateCluster>, DuplicateStats), AnalyticsError> {
        todo!()
    }

    pub async fn get_taxonomy_tree(
        &self,
        _root_path: Option<&str>,
    ) -> Result<Vec<serde_json::Value>, AnalyticsError> {
        todo!()
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
        fn _check<E: indexer::embeddings::EmbeddingProvider, V: indexer::qdrant::VectorRepository>(
        ) {
            assert_send_sync::<AnalyticsService<E, V>>();
        }
    }
}

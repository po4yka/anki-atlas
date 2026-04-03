use std::sync::Arc;
use tracing::instrument;

use crate::AnalyticsError;
use crate::coverage::{TopicCoverage, TopicGap, WeakNote};
use crate::duplicates::{DuplicateCluster, DuplicateDetector, DuplicateStats};
use crate::labeling::{LabelingStats, TopicLabeler};
use crate::repository::AnalyticsRepository;
use crate::taxonomy::Taxonomy;

/// Facade aggregating taxonomy, coverage, labeling, and duplicate detection.
pub struct AnalyticsService<E, V>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
{
    pub embedding: E,
    pub vector_repo: V,
    pub repository: Arc<dyn AnalyticsRepository>,
}

impl<E, V> AnalyticsService<E, V>
where
    E: indexer::embeddings::EmbeddingProvider,
    V: indexer::qdrant::VectorRepository,
{
    pub fn new(embedding: E, vector_repo: V, repository: Arc<dyn AnalyticsRepository>) -> Self {
        Self {
            embedding,
            vector_repo,
            repository,
        }
    }

    /// Load taxonomy from YAML (syncing to DB) or from DB.
    #[instrument(skip(self))]
    pub async fn load_taxonomy(
        &self,
        yaml_path: Option<&std::path::Path>,
    ) -> Result<Taxonomy, AnalyticsError> {
        self.repository.load_taxonomy(yaml_path).await
    }

    /// Label all notes with topics.
    #[instrument(skip(self))]
    pub async fn label_notes(
        &self,
        taxonomy: Option<&Taxonomy>,
        min_confidence: f32,
    ) -> Result<LabelingStats, AnalyticsError> {
        let owned_taxonomy;
        let tax = match taxonomy {
            Some(t) => t,
            None => {
                owned_taxonomy = self.repository.load_taxonomy(None).await?;
                &owned_taxonomy
            }
        };

        TopicLabeler::new(&self.embedding, Arc::clone(&self.repository))
            .label_notes(tax, min_confidence, 5, 100)
            .await
    }

    /// Get coverage metrics for a topic.
    #[instrument(skip(self))]
    pub async fn get_coverage(
        &self,
        topic_path: &str,
        include_subtree: bool,
    ) -> Result<Option<TopicCoverage>, AnalyticsError> {
        self.repository
            .get_topic_coverage(topic_path, include_subtree)
            .await
    }

    /// Find gaps in topic coverage under a root path.
    #[instrument(skip(self))]
    pub async fn get_gaps(
        &self,
        topic_path: &str,
        min_coverage: i64,
    ) -> Result<Vec<TopicGap>, AnalyticsError> {
        self.repository
            .get_topic_gaps(topic_path, min_coverage)
            .await
    }

    /// Get weak notes (high lapse rate) in a topic subtree.
    #[instrument(skip(self))]
    pub async fn get_weak_notes(
        &self,
        topic_path: &str,
        max_results: i64,
    ) -> Result<Vec<WeakNote>, AnalyticsError> {
        self.repository
            .get_weak_notes(topic_path, max_results, 0.1)
            .await
    }

    /// Find clusters of near-duplicate notes.
    #[instrument(skip(self))]
    pub async fn find_duplicates(
        &self,
        threshold: f64,
        max_clusters: usize,
        deck_filter: Option<&[String]>,
        tag_filter: Option<&[String]>,
    ) -> Result<(Vec<DuplicateCluster>, DuplicateStats), AnalyticsError> {
        DuplicateDetector::new(&self.vector_repo, Arc::clone(&self.repository))
            .find_duplicates(threshold, max_clusters, deck_filter, tag_filter)
            .await
    }

    /// Get coverage tree for all topics (optionally filtered by root path).
    #[instrument(skip(self))]
    pub async fn get_taxonomy_tree(
        &self,
        root_path: Option<&str>,
    ) -> Result<Vec<serde_json::Value>, AnalyticsError> {
        self.repository.get_coverage_tree(root_path).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    common::assert_send_sync!(
        AnalyticsError,
        crate::taxonomy::Topic,
        crate::taxonomy::Taxonomy,
        TopicCoverage,
        TopicGap,
        WeakNote,
        LabelingStats,
        DuplicateCluster,
        DuplicateStats,
    );

    #[test]
    fn analytics_service_is_generic() {
        // Verify the service compiles with mock types
        // (this is a compile-time check, the test body doesn't matter)
        fn _check<
            E: indexer::embeddings::EmbeddingProvider,
            V: indexer::qdrant::VectorRepository,
        >() {
            fn _assert<T: Send + Sync>() {}
            _assert::<AnalyticsService<E, V>>();
        }
    }
}

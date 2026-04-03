use std::path::Path;

use async_trait::async_trait;
use common::TopicId;
use sqlx::PgPool;

use crate::AnalyticsError;
use crate::coverage::{TopicCoverage, TopicGap, WeakNote};
use crate::taxonomy::Taxonomy;

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct ActiveNote {
    pub note_id: i64,
    pub normalized_text: String,
}

#[derive(Debug, Clone, sqlx::FromRow)]
pub struct NoteExcerptAndTags {
    pub excerpt: String,
    pub tags: Vec<String>,
}

#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait AnalyticsRepository: Send + Sync {
    async fn load_taxonomy(&self, yaml_path: Option<&Path>) -> Result<Taxonomy, AnalyticsError>;
    async fn get_topic_coverage(
        &self,
        topic_path: &str,
        include_subtree: bool,
    ) -> Result<Option<TopicCoverage>, AnalyticsError>;
    async fn get_topic_gaps(
        &self,
        topic_path: &str,
        min_coverage: i64,
    ) -> Result<Vec<TopicGap>, AnalyticsError>;
    async fn get_weak_notes(
        &self,
        topic_path: &str,
        max_results: i64,
        min_fail_rate: f64,
    ) -> Result<Vec<WeakNote>, AnalyticsError>;
    async fn get_coverage_tree(
        &self,
        root_path: Option<&str>,
    ) -> Result<Vec<serde_json::Value>, AnalyticsError>;
    async fn list_active_notes_batch(
        &self,
        batch_size: usize,
        offset: i64,
    ) -> Result<Vec<ActiveNote>, AnalyticsError>;
    async fn upsert_note_topic_assignment(
        &self,
        note_id: i64,
        topic_id: TopicId,
        confidence: f32,
        method: &str,
    ) -> Result<(), AnalyticsError>;
    async fn fetch_note_text(&self, note_id: i64) -> Result<String, AnalyticsError>;
    async fn fetch_active_note_ids(&self) -> Result<Vec<i64>, AnalyticsError>;
    async fn fetch_note_review_count(&self, note_id: i64) -> Result<i64, AnalyticsError>;
    async fn fetch_note_excerpt_and_tags(
        &self,
        note_id: i64,
    ) -> Result<NoteExcerptAndTags, AnalyticsError>;
    async fn fetch_note_deck_names(&self, note_id: i64) -> Result<Vec<String>, AnalyticsError>;
}

#[derive(Debug, Clone)]
pub struct SqlxAnalyticsRepository {
    pool: PgPool,
}

impl SqlxAnalyticsRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl AnalyticsRepository for SqlxAnalyticsRepository {
    async fn load_taxonomy(&self, yaml_path: Option<&Path>) -> Result<Taxonomy, AnalyticsError> {
        crate::taxonomy::load_taxonomy(&self.pool, yaml_path).await
    }

    async fn get_topic_coverage(
        &self,
        topic_path: &str,
        include_subtree: bool,
    ) -> Result<Option<TopicCoverage>, AnalyticsError> {
        crate::coverage::get_topic_coverage(&self.pool, topic_path, include_subtree).await
    }

    async fn get_topic_gaps(
        &self,
        topic_path: &str,
        min_coverage: i64,
    ) -> Result<Vec<TopicGap>, AnalyticsError> {
        crate::coverage::get_topic_gaps(&self.pool, topic_path, min_coverage).await
    }

    async fn get_weak_notes(
        &self,
        topic_path: &str,
        max_results: i64,
        min_fail_rate: f64,
    ) -> Result<Vec<WeakNote>, AnalyticsError> {
        crate::coverage::get_weak_notes(&self.pool, topic_path, max_results, min_fail_rate).await
    }

    async fn get_coverage_tree(
        &self,
        root_path: Option<&str>,
    ) -> Result<Vec<serde_json::Value>, AnalyticsError> {
        crate::coverage::get_coverage_tree(&self.pool, root_path).await
    }

    async fn list_active_notes_batch(
        &self,
        batch_size: usize,
        offset: i64,
    ) -> Result<Vec<ActiveNote>, AnalyticsError> {
        sqlx::query_as(
            "SELECT note_id, normalized_text FROM notes \
             WHERE deleted_at IS NULL \
             ORDER BY note_id LIMIT $1 OFFSET $2",
        )
        .bind(batch_size as i64)
        .bind(offset)
        .fetch_all(&self.pool)
        .await
        .map_err(Into::into)
    }

    async fn upsert_note_topic_assignment(
        &self,
        note_id: i64,
        topic_id: TopicId,
        confidence: f32,
        method: &str,
    ) -> Result<(), AnalyticsError> {
        sqlx::query(
            "INSERT INTO note_topics (note_id, topic_id, confidence, method) \
             VALUES ($1, $2, $3, $4) \
             ON CONFLICT (note_id, topic_id) DO UPDATE \
             SET confidence = $3, method = $4",
        )
        .bind(note_id)
        .bind(topic_id.0 as i32)
        .bind(confidence)
        .bind(method)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn fetch_note_text(&self, note_id: i64) -> Result<String, AnalyticsError> {
        let (text,): (String,) =
            sqlx::query_as("SELECT normalized_text FROM notes WHERE note_id = $1")
                .bind(note_id)
                .fetch_one(&self.pool)
                .await?;
        Ok(text)
    }

    async fn fetch_active_note_ids(&self) -> Result<Vec<i64>, AnalyticsError> {
        let note_ids: Vec<(i64,)> =
            sqlx::query_as("SELECT note_id FROM notes WHERE deleted_at IS NULL ORDER BY note_id")
                .fetch_all(&self.pool)
                .await?;
        Ok(note_ids.into_iter().map(|(note_id,)| note_id).collect())
    }

    async fn fetch_note_review_count(&self, note_id: i64) -> Result<i64, AnalyticsError> {
        let (review_count,): (i64,) =
            sqlx::query_as("SELECT COALESCE(SUM(c.reps), 0) FROM cards c WHERE c.note_id = $1")
                .bind(note_id)
                .fetch_one(&self.pool)
                .await?;
        Ok(review_count)
    }

    async fn fetch_note_excerpt_and_tags(
        &self,
        note_id: i64,
    ) -> Result<NoteExcerptAndTags, AnalyticsError> {
        sqlx::query_as(
            "SELECT LEFT(n.normalized_text, 200) AS excerpt, COALESCE(n.tags, '{}') AS tags \
             FROM notes n WHERE n.note_id = $1",
        )
        .bind(note_id)
        .fetch_one(&self.pool)
        .await
        .map_err(Into::into)
    }

    async fn fetch_note_deck_names(&self, note_id: i64) -> Result<Vec<String>, AnalyticsError> {
        let deck_rows: Vec<(String,)> = sqlx::query_as(
            "SELECT DISTINCT d.name FROM cards c \
             JOIN decks d ON d.deck_id = c.deck_id \
             WHERE c.note_id = $1",
        )
        .bind(note_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(deck_rows
            .into_iter()
            .map(|(deck_name,)| deck_name)
            .collect())
    }
}

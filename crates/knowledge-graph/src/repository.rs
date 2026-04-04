use std::str::FromStr;

use async_trait::async_trait;
use sqlx::PgPool;

use crate::error::KnowledgeGraphError;
use crate::models::{ConceptEdge, EdgeSource, EdgeType, TopicEdge};

/// Repository trait for knowledge graph CRUD and queries.
#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait KnowledgeGraphRepository: Send + Sync {
    /// Upsert concept-level edges. Returns count of rows affected.
    async fn upsert_concept_edges(
        &self,
        edges: &[ConceptEdge],
    ) -> Result<usize, KnowledgeGraphError>;

    /// Upsert topic-level edges. Returns count of rows affected.
    async fn upsert_topic_edges(&self, edges: &[TopicEdge]) -> Result<usize, KnowledgeGraphError>;

    /// Delete all edges created by a specific discovery source.
    async fn delete_edges_by_source(
        &self,
        source: EdgeSource,
    ) -> Result<usize, KnowledgeGraphError>;

    /// Get related notes for a given note, optionally filtered by edge type.
    async fn get_related_notes(
        &self,
        note_id: i64,
        edge_type: Option<EdgeType>,
        limit: usize,
    ) -> Result<Vec<ConceptEdge>, KnowledgeGraphError>;

    /// Get prerequisite notes (notes that should be learned first).
    async fn get_prerequisites(
        &self,
        note_id: i64,
    ) -> Result<Vec<ConceptEdge>, KnowledgeGraphError>;

    /// Get dependent notes (notes that depend on this one).
    async fn get_dependents(&self, note_id: i64) -> Result<Vec<ConceptEdge>, KnowledgeGraphError>;

    /// Get related topics, optionally filtered by edge type.
    async fn get_related_topics(
        &self,
        topic_id: i32,
        edge_type: Option<EdgeType>,
        limit: usize,
    ) -> Result<Vec<TopicEdge>, KnowledgeGraphError>;

    /// Get prerequisite topics for a given topic.
    async fn get_topic_prerequisites(
        &self,
        topic_id: i32,
    ) -> Result<Vec<TopicEdge>, KnowledgeGraphError>;

    /// Count of (concept_edges, topic_edges).
    async fn edge_count(&self) -> Result<(usize, usize), KnowledgeGraphError>;
}

/// PostgreSQL-backed knowledge graph repository.
pub struct SqlxKnowledgeGraphRepository {
    pool: PgPool,
}

impl SqlxKnowledgeGraphRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl KnowledgeGraphRepository for SqlxKnowledgeGraphRepository {
    async fn upsert_concept_edges(
        &self,
        edges: &[ConceptEdge],
    ) -> Result<usize, KnowledgeGraphError> {
        let mut count = 0usize;
        for edge in edges {
            let result = sqlx::query(
                "INSERT INTO concept_edges (source_note_id, target_note_id, edge_type, edge_source, weight)
                 VALUES ($1, $2, $3::edge_type, $4::edge_source, $5)
                 ON CONFLICT (source_note_id, target_note_id, edge_type) DO UPDATE
                 SET weight = EXCLUDED.weight, edge_source = EXCLUDED.edge_source",
            )
            .bind(edge.source_note_id)
            .bind(edge.target_note_id)
            .bind(edge.edge_type.to_string())
            .bind(edge.edge_source.to_string())
            .bind(edge.weight)
            .execute(&self.pool)
            .await?;
            count += result.rows_affected() as usize;
        }
        Ok(count)
    }

    async fn upsert_topic_edges(&self, edges: &[TopicEdge]) -> Result<usize, KnowledgeGraphError> {
        let mut count = 0usize;
        for edge in edges {
            let result = sqlx::query(
                "INSERT INTO topic_edges (source_topic_id, target_topic_id, edge_type, edge_source, weight)
                 VALUES ($1, $2, $3::edge_type, $4::edge_source, $5)
                 ON CONFLICT (source_topic_id, target_topic_id, edge_type) DO UPDATE
                 SET weight = EXCLUDED.weight, edge_source = EXCLUDED.edge_source",
            )
            .bind(edge.source_topic_id)
            .bind(edge.target_topic_id)
            .bind(edge.edge_type.to_string())
            .bind(edge.edge_source.to_string())
            .bind(edge.weight)
            .execute(&self.pool)
            .await?;
            count += result.rows_affected() as usize;
        }
        Ok(count)
    }

    async fn delete_edges_by_source(
        &self,
        source: EdgeSource,
    ) -> Result<usize, KnowledgeGraphError> {
        let source_str = source.to_string();
        let r1 = sqlx::query("DELETE FROM concept_edges WHERE edge_source = $1::edge_source")
            .bind(&source_str)
            .execute(&self.pool)
            .await?;
        let r2 = sqlx::query("DELETE FROM topic_edges WHERE edge_source = $1::edge_source")
            .bind(&source_str)
            .execute(&self.pool)
            .await?;
        Ok((r1.rows_affected() + r2.rows_affected()) as usize)
    }

    async fn get_related_notes(
        &self,
        note_id: i64,
        edge_type: Option<EdgeType>,
        limit: usize,
    ) -> Result<Vec<ConceptEdge>, KnowledgeGraphError> {
        let rows = match edge_type {
            Some(et) => sqlx::query_as::<_, ConceptEdgeRow>(
                "SELECT source_note_id, target_note_id, edge_type::text, edge_source::text, weight
                     FROM concept_edges
                     WHERE (source_note_id = $1 OR target_note_id = $1)
                       AND edge_type = $2::edge_type
                     ORDER BY weight DESC
                     LIMIT $3",
            )
            .bind(note_id)
            .bind(et.to_string())
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await?,
            None => sqlx::query_as::<_, ConceptEdgeRow>(
                "SELECT source_note_id, target_note_id, edge_type::text, edge_source::text, weight
                     FROM concept_edges
                     WHERE source_note_id = $1 OR target_note_id = $1
                     ORDER BY weight DESC
                     LIMIT $2",
            )
            .bind(note_id)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await?,
        };
        rows.into_iter().map(|r| r.into_edge()).collect()
    }

    async fn get_prerequisites(
        &self,
        note_id: i64,
    ) -> Result<Vec<ConceptEdge>, KnowledgeGraphError> {
        let rows = sqlx::query_as::<_, ConceptEdgeRow>(
            "SELECT source_note_id, target_note_id, edge_type::text, edge_source::text, weight
             FROM concept_edges
             WHERE target_note_id = $1 AND edge_type = 'prerequisite'
             ORDER BY weight DESC",
        )
        .bind(note_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(|r| r.into_edge()).collect()
    }

    async fn get_dependents(&self, note_id: i64) -> Result<Vec<ConceptEdge>, KnowledgeGraphError> {
        let rows = sqlx::query_as::<_, ConceptEdgeRow>(
            "SELECT source_note_id, target_note_id, edge_type::text, edge_source::text, weight
             FROM concept_edges
             WHERE source_note_id = $1 AND edge_type = 'prerequisite'
             ORDER BY weight DESC",
        )
        .bind(note_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(|r| r.into_edge()).collect()
    }

    async fn get_related_topics(
        &self,
        topic_id: i32,
        edge_type: Option<EdgeType>,
        limit: usize,
    ) -> Result<Vec<TopicEdge>, KnowledgeGraphError> {
        let rows = match edge_type {
            Some(et) => {
                sqlx::query_as::<_, TopicEdgeRow>(
                    "SELECT source_topic_id, target_topic_id, edge_type::text, edge_source::text, weight
                     FROM topic_edges
                     WHERE (source_topic_id = $1 OR target_topic_id = $1)
                       AND edge_type = $2::edge_type
                     ORDER BY weight DESC
                     LIMIT $3",
                )
                .bind(topic_id)
                .bind(et.to_string())
                .bind(limit as i64)
                .fetch_all(&self.pool)
                .await?
            }
            None => {
                sqlx::query_as::<_, TopicEdgeRow>(
                    "SELECT source_topic_id, target_topic_id, edge_type::text, edge_source::text, weight
                     FROM topic_edges
                     WHERE source_topic_id = $1 OR target_topic_id = $1
                     ORDER BY weight DESC
                     LIMIT $2",
                )
                .bind(topic_id)
                .bind(limit as i64)
                .fetch_all(&self.pool)
                .await?
            }
        };
        rows.into_iter().map(|r| r.into_edge()).collect()
    }

    async fn get_topic_prerequisites(
        &self,
        topic_id: i32,
    ) -> Result<Vec<TopicEdge>, KnowledgeGraphError> {
        let rows = sqlx::query_as::<_, TopicEdgeRow>(
            "SELECT source_topic_id, target_topic_id, edge_type::text, edge_source::text, weight
             FROM topic_edges
             WHERE target_topic_id = $1 AND edge_type = 'prerequisite'
             ORDER BY weight DESC",
        )
        .bind(topic_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(|r| r.into_edge()).collect()
    }

    async fn edge_count(&self) -> Result<(usize, usize), KnowledgeGraphError> {
        let concept: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM concept_edges")
            .fetch_one(&self.pool)
            .await?;
        let topic: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM topic_edges")
            .fetch_one(&self.pool)
            .await?;
        Ok((concept as usize, topic as usize))
    }
}

// --- Internal row types for sqlx mapping ---

#[derive(sqlx::FromRow)]
struct ConceptEdgeRow {
    source_note_id: i64,
    target_note_id: i64,
    edge_type: String,
    edge_source: String,
    weight: f32,
}

impl ConceptEdgeRow {
    fn into_edge(self) -> Result<ConceptEdge, KnowledgeGraphError> {
        Ok(ConceptEdge {
            source_note_id: self.source_note_id,
            target_note_id: self.target_note_id,
            edge_type: EdgeType::from_str(&self.edge_type).map_err(|_| {
                KnowledgeGraphError::Database(format!("unknown edge_type: {}", self.edge_type))
            })?,
            edge_source: EdgeSource::from_str(&self.edge_source).map_err(|_| {
                KnowledgeGraphError::Database(format!("unknown edge_source: {}", self.edge_source))
            })?,
            weight: self.weight,
        })
    }
}

#[derive(sqlx::FromRow)]
struct TopicEdgeRow {
    source_topic_id: i32,
    target_topic_id: i32,
    edge_type: String,
    edge_source: String,
    weight: f32,
}

impl TopicEdgeRow {
    fn into_edge(self) -> Result<TopicEdge, KnowledgeGraphError> {
        Ok(TopicEdge {
            source_topic_id: self.source_topic_id,
            target_topic_id: self.target_topic_id,
            edge_type: EdgeType::from_str(&self.edge_type).map_err(|_| {
                KnowledgeGraphError::Database(format!("unknown edge_type: {}", self.edge_type))
            })?,
            edge_source: EdgeSource::from_str(&self.edge_source).map_err(|_| {
                KnowledgeGraphError::Database(format!("unknown edge_source: {}", self.edge_source))
            })?,
            weight: self.weight,
        })
    }
}

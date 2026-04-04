use sqlx::PgPool;
use tracing::instrument;

use crate::error::KnowledgeGraphError;
use crate::models::{ConceptEdge, EdgeSource, EdgeType};
use crate::repository::KnowledgeGraphRepository;

/// Minimum shared tags to create a "related" edge.
const MIN_SHARED_TAGS: i64 = 3;

/// Discover related note edges based on tag co-occurrence.
///
/// Finds pairs of notes that share >= `MIN_SHARED_TAGS` tags and creates
/// bidirectional `Related` edges weighted by the fraction of shared tags.
#[instrument(skip_all)]
pub async fn discover_tag_cooccurrence_edges(
    pool: &PgPool,
    repo: &dyn KnowledgeGraphRepository,
) -> Result<usize, KnowledgeGraphError> {
    let rows = sqlx::query_as::<_, TagPairRow>(
        "WITH tag_pairs AS (
            SELECT a.note_id AS note_a, b.note_id AS note_b,
                   COUNT(*) AS shared,
                   GREATEST(array_length(a.tags, 1), array_length(b.tags, 1)) AS max_tags
            FROM notes a
            CROSS JOIN notes b
            CROSS JOIN UNNEST(a.tags) AS at
            CROSS JOIN UNNEST(b.tags) AS bt
            WHERE at = bt
              AND a.note_id < b.note_id
              AND a.deleted_at IS NULL
              AND b.deleted_at IS NULL
            GROUP BY a.note_id, b.note_id, a.tags, b.tags
            HAVING COUNT(*) >= $1
        )
        SELECT note_a, note_b, shared,
               (shared::real / NULLIF(max_tags, 0)::real) AS weight
        FROM tag_pairs
        ORDER BY weight DESC
        LIMIT 5000",
    )
    .bind(MIN_SHARED_TAGS)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Ok(0);
    }

    let edges: Vec<ConceptEdge> = rows
        .iter()
        .flat_map(|r| {
            let w = r.weight.unwrap_or(0.5);
            [
                ConceptEdge {
                    source_note_id: r.note_a,
                    target_note_id: r.note_b,
                    edge_type: EdgeType::Related,
                    edge_source: EdgeSource::TagCooccurrence,
                    weight: w,
                },
                ConceptEdge {
                    source_note_id: r.note_b,
                    target_note_id: r.note_a,
                    edge_type: EdgeType::Related,
                    edge_source: EdgeSource::TagCooccurrence,
                    weight: w,
                },
            ]
        })
        .collect();

    repo.upsert_concept_edges(&edges).await
}

#[derive(sqlx::FromRow)]
struct TagPairRow {
    note_a: i64,
    note_b: i64,
    #[allow(dead_code)]
    shared: i64,
    weight: Option<f32>,
}

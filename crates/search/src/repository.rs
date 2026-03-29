use std::collections::HashMap;

use async_trait::async_trait;
use sqlx::{FromRow, PgPool};

use crate::error::SearchError;
use crate::fts::{LexicalSearchResult, SearchFilters};
use crate::service::NoteDetail;

#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait SearchReadRepository: Send + Sync {
    async fn search_lexical(
        &self,
        query: &str,
        filters: Option<&SearchFilters>,
        limit: i64,
    ) -> Result<LexicalSearchResult, SearchError>;

    async fn get_note_details(&self, note_ids: &[i64]) -> Result<HashMap<i64, NoteDetail>, SearchError>;
}

#[derive(Debug, Clone)]
pub struct SqlxSearchReadRepository {
    pool: PgPool,
}

impl SqlxSearchReadRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[derive(Debug, FromRow)]
struct NoteDetailRow {
    note_id: i64,
    model_id: i64,
    normalized_text: String,
    tags: Vec<String>,
    deck_name: String,
    mature: bool,
    lapses: i32,
    reps: i32,
}

#[async_trait]
impl SearchReadRepository for SqlxSearchReadRepository {
    async fn search_lexical(
        &self,
        query: &str,
        filters: Option<&SearchFilters>,
        limit: i64,
    ) -> Result<LexicalSearchResult, SearchError> {
        crate::fts::search_lexical(&self.pool, query, filters, limit).await
    }

    async fn get_note_details(&self, note_ids: &[i64]) -> Result<HashMap<i64, NoteDetail>, SearchError> {
        if note_ids.is_empty() {
            return Ok(HashMap::new());
        }

        let rows = sqlx::query_as::<_, NoteDetailRow>(
            "SELECT n.note_id AS note_id, n.model_id AS model_id, n.normalized_text, \
             n.tags, \
             COALESCE(d.name, '') AS deck_name, \
             COALESCE(c.ivl >= 21, false) AS mature, \
             COALESCE(c.lapses, 0) AS lapses, \
             COALESCE(c.reps, 0) AS reps \
             FROM notes n \
             LEFT JOIN cards c ON c.note_id = n.note_id \
             LEFT JOIN decks d ON d.deck_id = c.deck_id \
             WHERE n.note_id = ANY($1)",
        )
        .bind(note_ids)
        .fetch_all(&self.pool)
        .await?;

        let mut map = HashMap::new();
        for row in rows {
            let entry = map.entry(row.note_id).or_insert_with(|| NoteDetail {
                note_id: row.note_id,
                model_id: row.model_id,
                normalized_text: row.normalized_text.clone(),
                tags: row.tags.clone(),
                deck_names: vec![],
                mature: row.mature,
                lapses: row.lapses,
                reps: row.reps,
            });
            entry.mature |= row.mature;
            entry.lapses = entry.lapses.max(row.lapses);
            entry.reps = entry.reps.max(row.reps);
            if !row.deck_name.is_empty() && !entry.deck_names.contains(&row.deck_name) {
                entry.deck_names.push(row.deck_name.clone());
            }
        }

        Ok(map)
    }
}

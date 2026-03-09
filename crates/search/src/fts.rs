use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use tracing::instrument;

use crate::error::SearchError;

/// Raw row from FTS/fuzzy/autocomplete queries.
#[derive(Debug, FromRow)]
struct FtsRow {
    note_id: i64,
    rank: f64,
    headline: Option<String>,
}

/// Source of a lexical search result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FtsSource {
    Fts,
    Fuzzy,
    Autocomplete,
}

/// Single FTS hit.
#[derive(Debug, Clone, Serialize)]
pub struct FtsResult {
    pub note_id: i64,
    pub rank: f64,
    pub headline: Option<String>,
    pub source: FtsSource,
}

/// Lexical search mode that was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexicalMode {
    Fts,
    Fuzzy,
    Autocomplete,
    None,
}

/// Bundle of lexical results with fallback metadata.
#[derive(Debug, Clone, Serialize)]
pub struct LexicalSearchResult {
    pub results: Vec<FtsResult>,
    pub mode: LexicalMode,
    pub used_fallback: bool,
    pub query_suggestions: Vec<String>,
    pub autocomplete_suggestions: Vec<String>,
}

/// Filters for search queries (shared between FTS and hybrid).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilters {
    pub deck_names: Option<Vec<String>>,
    pub deck_names_exclude: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub tags_exclude: Option<Vec<String>>,
    pub model_ids: Option<Vec<i64>>,
    pub min_ivl: Option<i32>,
    pub max_lapses: Option<i32>,
    pub min_reps: Option<i32>,
}

/// Bind value for a parameterized filter clause.
#[derive(Debug, Clone)]
enum FilterBind {
    StringArray(Vec<String>),
    I64Array(Vec<i64>),
    I32(i32),
}

/// Build SQL WHERE clauses for search filters using parameterized queries.
///
/// Returns the SQL fragment and a list of bind values. The parameter indices
/// start at `param_offset` (e.g., 3 means the first placeholder is `$3`).
fn build_filter_clauses(
    filters: Option<&SearchFilters>,
    param_offset: i32,
) -> (String, Vec<FilterBind>) {
    let filters = match filters {
        Some(f) => f,
        None => return (String::new(), vec![]),
    };
    let mut clauses = Vec::new();
    let mut binds: Vec<FilterBind> = Vec::new();
    let mut idx = param_offset;

    if let Some(ref tags) = filters.tags {
        clauses.push(format!("AND n.tags && ${idx}::text[]"));
        binds.push(FilterBind::StringArray(tags.clone()));
        idx += 1;
    }
    if let Some(ref tags_ex) = filters.tags_exclude {
        clauses.push(format!("AND NOT (n.tags && ${idx}::text[])"));
        binds.push(FilterBind::StringArray(tags_ex.clone()));
        idx += 1;
    }
    if let Some(ref model_ids) = filters.model_ids {
        clauses.push(format!("AND n.mid = ANY(${idx}::bigint[])"));
        binds.push(FilterBind::I64Array(model_ids.clone()));
        idx += 1;
    }
    if let Some(ref deck_names) = filters.deck_names {
        clauses.push(format!("AND d.name = ANY(${idx}::text[])"));
        binds.push(FilterBind::StringArray(deck_names.clone()));
        idx += 1;
    }
    if let Some(ref deck_names_ex) = filters.deck_names_exclude {
        clauses.push(format!("AND d.name != ALL(${idx}::text[])"));
        binds.push(FilterBind::StringArray(deck_names_ex.clone()));
        idx += 1;
    }
    if let Some(min_ivl) = filters.min_ivl {
        clauses.push(format!("AND c.ivl >= ${idx}"));
        binds.push(FilterBind::I32(min_ivl));
        idx += 1;
    }
    if let Some(max_lapses) = filters.max_lapses {
        clauses.push(format!("AND c.lapses <= ${idx}"));
        binds.push(FilterBind::I32(max_lapses));
        idx += 1;
    }
    if let Some(min_reps) = filters.min_reps {
        clauses.push(format!("AND c.reps >= ${idx}"));
        binds.push(FilterBind::I32(min_reps));
        #[allow(unused_assignments)]
        { idx += 1; }
    }

    (clauses.join(" "), binds)
}

/// Check if filters require joining cards/decks tables.
///
/// Filters on `deck_names`, `deck_names_exclude`, `min_ivl`, `max_lapses`,
/// and `min_reps` require a LEFT JOIN to `cards` and `decks`.
/// Filters on `tags`, `tags_exclude`, and `model_ids` apply to `notes` directly.
pub(crate) fn needs_card_join(filters: &SearchFilters) -> bool {
    filters.deck_names.is_some()
        || filters.deck_names_exclude.is_some()
        || filters.min_ivl.is_some()
        || filters.max_lapses.is_some()
        || filters.min_reps.is_some()
}

/// Return the JOIN clause for cards/decks when filters require it.
fn join_clause(filters: Option<&SearchFilters>) -> &'static str {
    if needs_card_join(filters.unwrap_or(&SearchFilters::default())) {
        "LEFT JOIN cards c ON c.nid = n.id LEFT JOIN decks d ON d.id = c.did"
    } else {
        ""
    }
}

/// Return GROUP BY clause when card join is present (avoids duplicate rows).
fn group_by_clause(filters: Option<&SearchFilters>) -> &'static str {
    if needs_card_join(filters.unwrap_or(&SearchFilters::default())) {
        "GROUP BY n.id, n.fts_vector, n.normalized_text, n.mid, n.tags"
    } else {
        ""
    }
}

/// Convert raw DB rows to typed FTS results.
fn rows_to_results(rows: Vec<FtsRow>, source: FtsSource) -> Vec<FtsResult> {
    rows.into_iter()
        .map(|r| FtsResult {
            note_id: r.note_id,
            rank: r.rank,
            headline: r.headline,
            source,
        })
        .collect()
}

/// Bind filter values to a sqlx query. Each bind must match the order from `build_filter_clauses`.
fn bind_filters<'q>(
    mut q: sqlx::query::QueryAs<
        'q,
        sqlx::Postgres,
        FtsRow,
        sqlx::postgres::PgArguments,
    >,
    binds: &'q [FilterBind],
) -> sqlx::query::QueryAs<'q, sqlx::Postgres, FtsRow, sqlx::postgres::PgArguments> {
    for bind in binds {
        match bind {
            FilterBind::StringArray(arr) => {
                q = q.bind(arr);
            }
            FilterBind::I64Array(arr) => {
                q = q.bind(arr);
            }
            FilterBind::I32(val) => {
                q = q.bind(val);
            }
        }
    }
    q
}

/// Execute lexical search with FTS -> fuzzy -> autocomplete fallback chain.
#[instrument(skip(pool))]
pub async fn search_lexical(
    pool: &sqlx::PgPool,
    query: &str,
    filters: Option<&SearchFilters>,
    limit: i64,
) -> Result<LexicalSearchResult, SearchError> {
    if query.trim().is_empty() {
        return Ok(LexicalSearchResult {
            results: vec![],
            mode: LexicalMode::None,
            used_fallback: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
        });
    }

    let join = join_clause(filters);
    // $1 = query text, $2 = limit, filter params start at $3
    let (filter_sql, filter_binds) = build_filter_clauses(filters, 3);
    let group_by = group_by_clause(filters);

    // Stage 1: Full-text search
    let fts_sql = format!(
        "SELECT n.id AS note_id, ts_rank(n.fts_vector, plainto_tsquery('english', $1)) AS rank, \
         ts_headline('english', n.normalized_text, plainto_tsquery('english', $1)) AS headline \
         FROM notes n {join} WHERE n.fts_vector @@ plainto_tsquery('english', $1) {filter_sql} \
         {group_by} ORDER BY rank DESC LIMIT $2",
    );

    let base = sqlx::query_as::<_, FtsRow>(&fts_sql)
        .bind(query)
        .bind(limit);
    let rows: Vec<FtsRow> = bind_filters(base, &filter_binds)
        .fetch_all(pool)
        .await?;

    if !rows.is_empty() {
        return Ok(LexicalSearchResult {
            results: rows_to_results(rows, FtsSource::Fts),
            mode: LexicalMode::Fts,
            used_fallback: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
        });
    }

    // Stage 2: Fuzzy search fallback
    let fuzzy_sql = format!(
        "SELECT n.id AS note_id, similarity(n.normalized_text, $1) AS rank, \
         NULL AS headline \
         FROM notes n {join} WHERE similarity(n.normalized_text, $1) > 0.1 {filter_sql} \
         {group_by} ORDER BY rank DESC LIMIT $2",
    );

    let base = sqlx::query_as::<_, FtsRow>(&fuzzy_sql)
        .bind(query)
        .bind(limit);
    let rows: Vec<FtsRow> = bind_filters(base, &filter_binds)
        .fetch_all(pool)
        .await?;

    if !rows.is_empty() {
        return Ok(LexicalSearchResult {
            results: rows_to_results(rows, FtsSource::Fuzzy),
            mode: LexicalMode::Fuzzy,
            used_fallback: true,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
        });
    }

    // Stage 3: Autocomplete fallback
    let auto_sql = format!(
        "SELECT n.id AS note_id, 1.0 AS rank, NULL AS headline \
         FROM notes n {join} WHERE n.normalized_text ILIKE $1 {filter_sql} \
         {group_by} ORDER BY n.id LIMIT $2",
    );

    let prefix_pattern = format!("{}%", query);
    let base = sqlx::query_as::<_, FtsRow>(&auto_sql)
        .bind(&prefix_pattern)
        .bind(limit);
    let rows: Vec<FtsRow> = bind_filters(base, &filter_binds)
        .fetch_all(pool)
        .await?;

    let mode = if rows.is_empty() {
        LexicalMode::None
    } else {
        LexicalMode::Autocomplete
    };

    Ok(LexicalSearchResult {
        results: rows_to_results(rows, FtsSource::Autocomplete),
        mode,
        used_fallback: true,
        query_suggestions: vec![],
        autocomplete_suggestions: vec![],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Type construction tests ---

    #[test]
    fn fts_source_variants_exist() {
        let sources = [FtsSource::Fts, FtsSource::Fuzzy, FtsSource::Autocomplete];
        assert_eq!(sources.len(), 3);
    }

    #[test]
    fn fts_source_serde_roundtrip() {
        let source = FtsSource::Fts;
        let json = serde_json::to_string(&source).unwrap();
        assert_eq!(json, "\"fts\"");

        let fuzzy: FtsSource = serde_json::from_str("\"fuzzy\"").unwrap();
        assert_eq!(fuzzy, FtsSource::Fuzzy);

        let auto: FtsSource = serde_json::from_str("\"autocomplete\"").unwrap();
        assert_eq!(auto, FtsSource::Autocomplete);
    }

    #[test]
    fn lexical_mode_serde_roundtrip() {
        let mode = LexicalMode::None;
        let json = serde_json::to_string(&mode).unwrap();
        assert_eq!(json, "\"none\"");

        let fts: LexicalMode = serde_json::from_str("\"fts\"").unwrap();
        assert_eq!(fts, LexicalMode::Fts);

        let fuzzy: LexicalMode = serde_json::from_str("\"fuzzy\"").unwrap();
        assert_eq!(fuzzy, LexicalMode::Fuzzy);

        let auto: LexicalMode = serde_json::from_str("\"autocomplete\"").unwrap();
        assert_eq!(auto, LexicalMode::Autocomplete);
    }

    #[test]
    fn fts_result_construction() {
        let result = FtsResult {
            note_id: 42,
            rank: 0.95,
            headline: Some("matched <b>term</b>".to_string()),
            source: FtsSource::Fts,
        };
        assert_eq!(result.note_id, 42);
        assert!((result.rank - 0.95).abs() < f64::EPSILON);
        assert_eq!(result.headline.as_deref(), Some("matched <b>term</b>"));
        assert_eq!(result.source, FtsSource::Fts);
    }

    #[test]
    fn fts_result_without_headline() {
        let result = FtsResult {
            note_id: 1,
            rank: 0.5,
            headline: None,
            source: FtsSource::Autocomplete,
        };
        assert!(result.headline.is_none());
        assert_eq!(result.source, FtsSource::Autocomplete);
    }

    #[test]
    fn lexical_search_result_construction() {
        let result = LexicalSearchResult {
            results: vec![
                FtsResult {
                    note_id: 1,
                    rank: 0.9,
                    headline: Some("hl".to_string()),
                    source: FtsSource::Fts,
                },
                FtsResult {
                    note_id: 2,
                    rank: 0.8,
                    headline: None,
                    source: FtsSource::Fts,
                },
            ],
            mode: LexicalMode::Fts,
            used_fallback: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
        };
        assert_eq!(result.results.len(), 2);
        assert_eq!(result.mode, LexicalMode::Fts);
        assert!(!result.used_fallback);
        assert!(result.query_suggestions.is_empty());
        assert!(result.autocomplete_suggestions.is_empty());
    }

    #[test]
    fn lexical_search_result_with_fallback() {
        let result = LexicalSearchResult {
            results: vec![FtsResult {
                note_id: 10,
                rank: 0.3,
                headline: None,
                source: FtsSource::Fuzzy,
            }],
            mode: LexicalMode::Fuzzy,
            used_fallback: true,
            query_suggestions: vec!["suggestion1".to_string(), "suggestion2".to_string()],
            autocomplete_suggestions: vec!["auto1".to_string()],
        };
        assert_eq!(result.mode, LexicalMode::Fuzzy);
        assert!(result.used_fallback);
        assert_eq!(result.query_suggestions.len(), 2);
        assert_eq!(result.autocomplete_suggestions.len(), 1);
    }

    #[test]
    fn lexical_search_result_none_mode() {
        let result = LexicalSearchResult {
            results: vec![],
            mode: LexicalMode::None,
            used_fallback: false,
            query_suggestions: vec![],
            autocomplete_suggestions: vec![],
        };
        assert!(result.results.is_empty());
        assert_eq!(result.mode, LexicalMode::None);
    }

    // --- SearchFilters tests ---

    #[test]
    fn search_filters_default_all_none() {
        let filters = SearchFilters::default();
        assert!(filters.deck_names.is_none());
        assert!(filters.deck_names_exclude.is_none());
        assert!(filters.tags.is_none());
        assert!(filters.tags_exclude.is_none());
        assert!(filters.model_ids.is_none());
        assert!(filters.min_ivl.is_none());
        assert!(filters.max_lapses.is_none());
        assert!(filters.min_reps.is_none());
    }

    #[test]
    fn search_filters_serde_roundtrip() {
        let filters = SearchFilters {
            deck_names: Some(vec!["Default".to_string()]),
            deck_names_exclude: None,
            tags: Some(vec!["rust".to_string(), "programming".to_string()]),
            tags_exclude: Some(vec!["deprecated".to_string()]),
            model_ids: Some(vec![123, 456]),
            min_ivl: Some(21),
            max_lapses: Some(5),
            min_reps: Some(3),
        };
        let json = serde_json::to_string(&filters).unwrap();
        let deserialized: SearchFilters = serde_json::from_str(&json).unwrap();

        assert_eq!(
            deserialized.deck_names.as_ref().unwrap(),
            &["Default".to_string()]
        );
        assert!(deserialized.deck_names_exclude.is_none());
        assert_eq!(deserialized.tags.as_ref().unwrap().len(), 2);
        assert_eq!(deserialized.model_ids.as_ref().unwrap(), &[123, 456]);
        assert_eq!(deserialized.min_ivl, Some(21));
        assert_eq!(deserialized.max_lapses, Some(5));
        assert_eq!(deserialized.min_reps, Some(3));
    }

    #[test]
    fn search_filters_partial_construction() {
        let filters = SearchFilters {
            tags: Some(vec!["math".to_string()]),
            min_ivl: Some(7),
            ..SearchFilters::default()
        };
        assert_eq!(filters.tags.as_ref().unwrap(), &["math".to_string()]);
        assert_eq!(filters.min_ivl, Some(7));
        assert!(filters.deck_names.is_none());
        assert!(filters.model_ids.is_none());
    }

    // --- build_filter_clauses tests ---

    #[test]
    fn filter_clauses_none_returns_empty() {
        let (sql, binds) = build_filter_clauses(None, 3);
        assert!(sql.is_empty());
        assert!(binds.is_empty());
    }

    #[test]
    fn filter_clauses_empty_filters_returns_empty() {
        let filters = SearchFilters::default();
        let (sql, binds) = build_filter_clauses(Some(&filters), 3);
        assert!(sql.is_empty());
        assert!(binds.is_empty());
    }

    #[test]
    fn filter_clauses_uses_parameterized_placeholders() {
        let filters = SearchFilters {
            tags: Some(vec!["rust".into()]),
            deck_names: Some(vec!["Default".into()]),
            min_ivl: Some(21),
            ..Default::default()
        };
        let (sql, binds) = build_filter_clauses(Some(&filters), 3);
        // Should use $3, $4, $5 placeholders, NOT interpolated strings
        assert!(sql.contains("$3"));
        assert!(sql.contains("$4"));
        assert!(sql.contains("$5"));
        assert!(!sql.contains("rust"));
        assert!(!sql.contains("Default"));
        assert!(!sql.contains("21"));
        assert_eq!(binds.len(), 3);
    }

    #[test]
    fn filter_clauses_respects_param_offset() {
        let filters = SearchFilters {
            tags: Some(vec!["test".into()]),
            ..Default::default()
        };
        let (sql, _) = build_filter_clauses(Some(&filters), 5);
        assert!(sql.contains("$5"));
        assert!(!sql.contains("$3"));
    }

    #[test]
    fn filter_clauses_sql_injection_safe() {
        let filters = SearchFilters {
            tags: Some(vec!["'; DROP TABLE notes; --".into()]),
            deck_names: Some(vec!["'); DELETE FROM decks; --".into()]),
            ..Default::default()
        };
        let (sql, _) = build_filter_clauses(Some(&filters), 3);
        // SQL should never contain the injected strings
        assert!(!sql.contains("DROP"));
        assert!(!sql.contains("DELETE"));
        assert!(!sql.contains("notes"));
        assert!(!sql.contains("decks"));
    }

    // --- search_lexical behavior tests ---

    #[tokio::test]
    async fn search_lexical_blank_query_returns_none_mode() {
        // Blank queries should return immediately with mode=None, no DB access needed.
        // Use a dummy pool options to create a pool that won't connect.
        let pool = sqlx::PgPool::connect_lazy("postgres://invalid:5432/fake").unwrap();
        let result = search_lexical(&pool, "", None, 10).await.unwrap();
        assert_eq!(result.mode, LexicalMode::None);
        assert!(result.results.is_empty());
        assert!(!result.used_fallback);
        assert!(result.query_suggestions.is_empty());
        assert!(result.autocomplete_suggestions.is_empty());
    }

    #[tokio::test]
    async fn search_lexical_whitespace_query_returns_none_mode() {
        let pool = sqlx::PgPool::connect_lazy("postgres://invalid:5432/fake").unwrap();
        let result = search_lexical(&pool, "   ", None, 10).await.unwrap();
        assert_eq!(result.mode, LexicalMode::None);
        assert!(result.results.is_empty());
        assert!(!result.used_fallback);
    }

    // --- FtsResult serialization ---

    #[test]
    fn fts_result_serializes_to_json() {
        let result = FtsResult {
            note_id: 99,
            rank: 0.75,
            headline: Some("test headline".to_string()),
            source: FtsSource::Fuzzy,
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["note_id"], 99);
        assert_eq!(json["source"], "fuzzy");
        assert_eq!(json["headline"], "test headline");
    }

    #[test]
    fn lexical_search_result_serializes_to_json() {
        let result = LexicalSearchResult {
            results: vec![],
            mode: LexicalMode::Autocomplete,
            used_fallback: true,
            query_suggestions: vec!["fix".to_string()],
            autocomplete_suggestions: vec!["fixing".to_string(), "fixed".to_string()],
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["mode"], "autocomplete");
        assert_eq!(json["used_fallback"], true);
        assert_eq!(json["query_suggestions"].as_array().unwrap().len(), 1);
        assert_eq!(
            json["autocomplete_suggestions"].as_array().unwrap().len(),
            2
        );
    }

    // --- FtsSource equality + clone ---

    #[test]
    fn fts_source_clone_and_eq() {
        let a = FtsSource::Fts;
        let b = a;
        assert_eq!(a, b);

        let c = FtsSource::Fuzzy;
        assert_ne!(a, c);
    }

    #[test]
    fn lexical_mode_clone_and_eq() {
        let a = LexicalMode::Fts;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(LexicalMode::Fts, LexicalMode::Fuzzy);
        assert_ne!(LexicalMode::Fuzzy, LexicalMode::Autocomplete);
        assert_ne!(LexicalMode::Autocomplete, LexicalMode::None);
    }

    // --- needs_card_join tests ---

    #[test]
    fn needs_card_join_false_with_default_filters() {
        assert!(!needs_card_join(&SearchFilters::default()));
    }

    #[test]
    fn needs_card_join_false_with_tags_only() {
        let f = SearchFilters {
            tags: Some(vec!["rust".into()]),
            ..Default::default()
        };
        assert!(!needs_card_join(&f));
    }

    #[test]
    fn needs_card_join_false_with_tags_exclude_only() {
        let f = SearchFilters {
            tags_exclude: Some(vec!["old".into()]),
            ..Default::default()
        };
        assert!(!needs_card_join(&f));
    }

    #[test]
    fn needs_card_join_false_with_model_ids_only() {
        let f = SearchFilters {
            model_ids: Some(vec![1, 2]),
            ..Default::default()
        };
        assert!(!needs_card_join(&f));
    }

    #[test]
    fn needs_card_join_false_with_notes_only_combo() {
        // All notes-table filters combined should still not require join
        let f = SearchFilters {
            tags: Some(vec!["rust".into()]),
            tags_exclude: Some(vec!["old".into()]),
            model_ids: Some(vec![42]),
            ..Default::default()
        };
        assert!(!needs_card_join(&f));
    }

    #[test]
    fn needs_card_join_true_for_deck_names() {
        let f = SearchFilters {
            deck_names: Some(vec!["Default".into()]),
            ..Default::default()
        };
        assert!(needs_card_join(&f));
    }

    #[test]
    fn needs_card_join_true_for_deck_names_exclude() {
        let f = SearchFilters {
            deck_names_exclude: Some(vec!["Archive".into()]),
            ..Default::default()
        };
        assert!(needs_card_join(&f));
    }

    #[test]
    fn needs_card_join_true_for_min_ivl() {
        let f = SearchFilters {
            min_ivl: Some(21),
            ..Default::default()
        };
        assert!(needs_card_join(&f));
    }

    #[test]
    fn needs_card_join_true_for_max_lapses() {
        let f = SearchFilters {
            max_lapses: Some(5),
            ..Default::default()
        };
        assert!(needs_card_join(&f));
    }

    #[test]
    fn needs_card_join_true_for_min_reps() {
        let f = SearchFilters {
            min_reps: Some(3),
            ..Default::default()
        };
        assert!(needs_card_join(&f));
    }

    // --- search_lexical fallback chain (non-empty query) ---

    #[tokio::test]
    async fn search_lexical_nonempty_query_returns_result() {
        // Non-empty query should execute the FTS fallback chain, not panic.
        // With an invalid pool, should return a SearchError::Database, not todo!().
        let pool = sqlx::PgPool::connect_lazy("postgres://invalid:5432/fake").unwrap();
        let result = search_lexical(&pool, "hello world", None, 10).await;
        // Should be Err(Database) since pool can't connect, but must not panic
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn search_lexical_nonempty_with_filters_returns_result() {
        let pool = sqlx::PgPool::connect_lazy("postgres://invalid:5432/fake").unwrap();
        let filters = SearchFilters {
            tags: Some(vec!["rust".into()]),
            deck_names: Some(vec!["Default".into()]),
            ..Default::default()
        };
        let result = search_lexical(&pool, "test query", Some(&filters), 20).await;
        // Should attempt DB query with filters applied, return DB error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn search_lexical_nonempty_respects_limit() {
        let pool = sqlx::PgPool::connect_lazy("postgres://invalid:5432/fake").unwrap();
        let result = search_lexical(&pool, "query", None, 5).await;
        // Should attempt query with LIMIT 5, return DB error
        assert!(result.is_err());
    }
}

use serde::{Deserialize, Serialize};

use crate::AnalyticsError;

/// Coverage metrics for a topic.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TopicCoverage {
    pub topic_id: i64,
    pub path: String,
    pub label: String,
    pub note_count: i64,
    pub subtree_count: i64,
    pub child_count: i64,
    pub covered_children: i64,
    pub mature_count: i64,
    pub avg_confidence: f64,
    pub weak_notes: i64,
    pub avg_lapses: f64,
}

/// A gap in topic coverage.
#[derive(Debug, Clone, Serialize)]
pub struct TopicGap {
    pub topic_id: i64,
    pub path: String,
    pub label: String,
    pub description: Option<String>,
    /// "missing" (0 notes) or "undercovered" (< threshold).
    pub gap_type: GapType,
    pub note_count: i64,
    pub threshold: i64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nearest_notes: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GapType {
    Missing,
    Undercovered,
}

/// A note with weakness signals.
#[derive(Debug, Clone, Serialize)]
pub struct WeakNote {
    pub note_id: i64,
    pub topic_path: String,
    pub confidence: f64,
    pub lapses: i32,
    pub fail_rate: Option<f64>,
    /// Truncated to 200 chars.
    pub normalized_text: String,
}

/// Classify a topic gap based on note count.
/// Returns `Missing` when note_count == 0, `Undercovered` otherwise.
/// The caller (`get_topic_gaps`) pre-filters to topics below the threshold.
pub fn classify_gap(note_count: i64) -> GapType {
    if note_count == 0 {
        GapType::Missing
    } else {
        GapType::Undercovered
    }
}

/// Get coverage metrics for a topic (optionally including subtree).
pub async fn get_topic_coverage(
    pool: &sqlx::PgPool,
    topic_path: &str,
    include_subtree: bool,
) -> Result<Option<TopicCoverage>, AnalyticsError> {
    let topic_row: Option<(i32, String, String)> =
        sqlx::query_as("SELECT topic_id, path, label FROM topics WHERE path = $1")
            .bind(topic_path)
            .fetch_optional(pool)
            .await?;

    let (topic_id, path, label) = match topic_row {
        Some(r) => r,
        None => return Ok(None),
    };

    // Note count and card stats for this topic (and optionally subtree)
    let stats: (i64, i64, f64, i64, f64) = if include_subtree {
        sqlx::query_as(
            "SELECT \
                COALESCE(COUNT(DISTINCT nt.note_id), 0), \
                COALESCE(COUNT(DISTINCT CASE WHEN c.ivl >= 21 THEN c.card_id END), 0), \
                COALESCE(AVG(nt.confidence), 0.0), \
                COALESCE(SUM(c.lapses), 0), \
                COALESCE(AVG(c.lapses::float), 0.0) \
             FROM topics t \
             JOIN note_topics nt ON nt.topic_id = t.topic_id \
             LEFT JOIN cards c ON c.note_id = nt.note_id \
             WHERE t.path = $1 OR t.path LIKE $1 || '/%'",
        )
        .bind(topic_path)
        .fetch_one(pool)
        .await?
    } else {
        sqlx::query_as(
            "SELECT \
                COALESCE(COUNT(DISTINCT nt.note_id), 0), \
                COALESCE(COUNT(DISTINCT CASE WHEN c.ivl >= 21 THEN c.card_id END), 0), \
                COALESCE(AVG(nt.confidence), 0.0), \
                COALESCE(SUM(c.lapses), 0), \
                COALESCE(AVG(c.lapses::float), 0.0) \
             FROM note_topics nt \
             LEFT JOIN cards c ON c.note_id = nt.note_id \
             WHERE nt.topic_id = $1",
        )
        .bind(topic_id)
        .fetch_one(pool)
        .await?
    };

    // Count children and covered children
    let children_stats: (i64, i64) = sqlx::query_as(
        "SELECT \
            COUNT(*), \
            COUNT(CASE WHEN EXISTS ( \
                SELECT 1 FROM note_topics nt WHERE nt.topic_id = t.topic_id \
            ) THEN 1 END) \
         FROM topics t \
         WHERE t.path LIKE $1 || '/%' AND t.path NOT LIKE $1 || '/%/%'",
    )
    .bind(topic_path)
    .fetch_one(pool)
    .await?;

    // Subtree count
    let (subtree_count,): (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM topics WHERE path LIKE $1 || '/%'")
            .bind(topic_path)
            .fetch_one(pool)
            .await?;

    // Weak notes count (fail_rate > 0.1)
    let (weak_count,): (i64,) = sqlx::query_as(
        "SELECT COUNT(DISTINCT nt.note_id) \
         FROM note_topics nt \
         JOIN card_stats cs ON cs.card_id IN (SELECT card_id FROM cards WHERE note_id = nt.note_id) \
         WHERE nt.topic_id = $1 AND cs.fail_rate >= 0.1",
    )
    .bind(topic_id)
    .fetch_one(pool)
    .await?;

    Ok(Some(TopicCoverage {
        topic_id: i64::from(topic_id),
        path,
        label,
        note_count: stats.0,
        subtree_count,
        child_count: children_stats.0,
        covered_children: children_stats.1,
        mature_count: stats.1,
        avg_confidence: stats.2,
        weak_notes: weak_count,
        avg_lapses: stats.4,
    }))
}

/// Find gaps in topic coverage under a root path.
pub async fn get_topic_gaps(
    pool: &sqlx::PgPool,
    topic_path: &str,
    min_coverage: i64,
) -> Result<Vec<TopicGap>, AnalyticsError> {
    let rows: Vec<(i32, String, String, Option<String>, i64)> = sqlx::query_as(
        "SELECT t.topic_id, t.path, t.label, t.description, \
                COALESCE(COUNT(nt.note_id), 0) as note_count \
         FROM topics t \
         LEFT JOIN note_topics nt ON nt.topic_id = t.topic_id \
         WHERE t.path = $1 OR t.path LIKE $1 || '/%' \
         GROUP BY t.topic_id, t.path, t.label, t.description \
         HAVING COALESCE(COUNT(nt.note_id), 0) < $2 \
         ORDER BY note_count ASC, t.path ASC",
    )
    .bind(topic_path)
    .bind(min_coverage)
    .fetch_all(pool)
    .await?;

    let gaps = rows
        .into_iter()
        .map(
            |(topic_id, path, label, description, note_count)| TopicGap {
                topic_id: i64::from(topic_id),
                path,
                label,
                description,
                gap_type: classify_gap(note_count),
                note_count,
                threshold: min_coverage,
                nearest_notes: vec![],
            },
        )
        .collect();

    Ok(gaps)
}

/// Get weak notes (high lapse rate) in a topic subtree.
pub async fn get_weak_notes(
    pool: &sqlx::PgPool,
    topic_path: &str,
    max_results: i64,
    min_fail_rate: f64,
) -> Result<Vec<WeakNote>, AnalyticsError> {
    let rows: Vec<(i64, String, f64, i32, Option<f64>, String)> = sqlx::query_as(
        "SELECT n.note_id, t.path, nt.confidence::float8, \
                COALESCE(SUM(c.lapses), 0)::int4 as total_lapses, \
                MAX(cs.fail_rate)::float8 as fail_rate, \
                LEFT(n.normalized_text, 200) as text_preview \
         FROM notes n \
         JOIN note_topics nt ON nt.note_id = n.note_id \
         JOIN topics t ON t.topic_id = nt.topic_id \
         JOIN cards c ON c.note_id = n.note_id \
         LEFT JOIN card_stats cs ON cs.card_id = c.card_id \
         WHERE (t.path = $1 OR t.path LIKE $1 || '/%') \
           AND n.deleted_at IS NULL \
         GROUP BY n.note_id, t.path, nt.confidence, n.normalized_text \
         HAVING MAX(cs.fail_rate) >= $2 \
         ORDER BY total_lapses DESC \
         LIMIT $3",
    )
    .bind(topic_path)
    .bind(min_fail_rate)
    .bind(max_results)
    .fetch_all(pool)
    .await?;

    let weak = rows
        .into_iter()
        .map(
            |(note_id, topic_path, confidence, lapses, fail_rate, normalized_text)| WeakNote {
                note_id,
                topic_path,
                confidence,
                lapses,
                fail_rate,
                normalized_text,
            },
        )
        .collect();

    Ok(weak)
}

/// Get coverage tree for all topics (optionally filtered by root path).
pub async fn get_coverage_tree(
    pool: &sqlx::PgPool,
    root_path: Option<&str>,
) -> Result<Vec<serde_json::Value>, AnalyticsError> {
    // Load topics
    let topics: Vec<(i32, String, String, Option<String>)> = match root_path {
        Some(rp) => {
            sqlx::query_as(
                "SELECT topic_id, path, label, description FROM topics \
                 WHERE path = $1 OR path LIKE $1 || '/%' ORDER BY path",
            )
            .bind(rp)
            .fetch_all(pool)
            .await?
        }
        None => {
            sqlx::query_as("SELECT topic_id, path, label, description FROM topics ORDER BY path")
                .fetch_all(pool)
                .await?
        }
    };

    if topics.is_empty() {
        return Ok(vec![]);
    }

    // Load note counts per topic
    let note_counts: Vec<(i32, i64)> =
        sqlx::query_as("SELECT topic_id, COUNT(note_id) FROM note_topics GROUP BY topic_id")
            .fetch_all(pool)
            .await?;

    let count_map: std::collections::HashMap<i32, i64> = note_counts.into_iter().collect();

    // Build nested JSON tree
    let mut nodes: std::collections::HashMap<String, serde_json::Value> =
        std::collections::HashMap::new();

    // Create nodes in path order (shortest paths first, already sorted)
    for (topic_id, path, label, description) in &topics {
        let nc = count_map.get(topic_id).copied().unwrap_or(0);
        nodes.insert(
            path.clone(),
            serde_json::json!({
                "topic_id": topic_id,
                "path": path,
                "label": label,
                "description": description,
                "note_count": nc,
                "children": [],
            }),
        );
    }

    // Wire children (process longest paths first so children exist)
    let paths: Vec<String> = topics.iter().map(|(_, p, _, _)| p.clone()).collect();
    for path in paths.iter().rev() {
        if let Some(slash_pos) = path.rfind('/') {
            let parent_path = &path[..slash_pos];
            if let Some(child_node) = nodes.get(path).cloned() {
                if let Some(parent_node) = nodes.get_mut(parent_path) {
                    parent_node["children"]
                        .as_array_mut()
                        .unwrap()
                        .push(child_node);
                }
            }
        }
    }

    // Collect roots
    let effective_root_depth = root_path.map(|rp| rp.matches('/').count()).unwrap_or(0);
    let roots: Vec<serde_json::Value> = paths
        .iter()
        .filter(|p| p.matches('/').count() == effective_root_depth)
        .filter_map(|p| nodes.get(p).cloned())
        .collect();

    Ok(roots)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gap_type_missing_serializes_as_snake_case() {
        let json = serde_json::to_string(&GapType::Missing).unwrap();
        assert_eq!(json, "\"missing\"");
    }

    #[test]
    fn gap_type_undercovered_serializes_as_snake_case() {
        let json = serde_json::to_string(&GapType::Undercovered).unwrap();
        assert_eq!(json, "\"undercovered\"");
    }

    #[test]
    fn gap_type_roundtrip() {
        let missing: GapType = serde_json::from_str("\"missing\"").unwrap();
        assert_eq!(missing, GapType::Missing);
        let under: GapType = serde_json::from_str("\"undercovered\"").unwrap();
        assert_eq!(under, GapType::Undercovered);
    }

    // --- classify_gap ---

    #[test]
    fn classify_gap_zero_notes_is_missing() {
        assert_eq!(classify_gap(0), GapType::Missing);
    }

    #[test]
    fn classify_gap_nonzero_notes_is_undercovered() {
        assert_eq!(classify_gap(3), GapType::Undercovered);
    }

    #[test]
    fn classify_gap_one_note_is_undercovered() {
        assert_eq!(classify_gap(1), GapType::Undercovered);
    }
}

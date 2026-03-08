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

/// Classify a topic gap based on note count vs threshold.
/// Returns `Missing` when note_count == 0, `Undercovered` otherwise.
#[allow(dead_code)]
pub fn classify_gap(_note_count: i64, _threshold: i64) -> GapType {
    todo!()
}

/// Get coverage metrics for a topic (optionally including subtree).
pub async fn get_topic_coverage(
    _pool: &sqlx::PgPool,
    _topic_path: &str,
    _include_subtree: bool,
) -> Result<Option<TopicCoverage>, AnalyticsError> {
    todo!()
}

/// Find gaps in topic coverage under a root path.
pub async fn get_topic_gaps(
    _pool: &sqlx::PgPool,
    _topic_path: &str,
    _min_coverage: i64,
) -> Result<Vec<TopicGap>, AnalyticsError> {
    todo!()
}

/// Get weak notes (high lapse rate) in a topic subtree.
pub async fn get_weak_notes(
    _pool: &sqlx::PgPool,
    _topic_path: &str,
    _max_results: i64,
    _min_fail_rate: f64,
) -> Result<Vec<WeakNote>, AnalyticsError> {
    todo!()
}

/// Get coverage tree for all topics (optionally filtered by root path).
pub async fn get_coverage_tree(
    _pool: &sqlx::PgPool,
    _root_path: Option<&str>,
) -> Result<Vec<serde_json::Value>, AnalyticsError> {
    todo!()
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
        assert_eq!(classify_gap(0, 5), GapType::Missing);
    }

    #[test]
    fn classify_gap_below_threshold_is_undercovered() {
        assert_eq!(classify_gap(3, 5), GapType::Undercovered);
    }

    #[test]
    fn classify_gap_one_note_is_undercovered() {
        assert_eq!(classify_gap(1, 10), GapType::Undercovered);
    }
}

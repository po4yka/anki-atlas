use common::types::NoteId;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct TaxonomyLoadSummary {
    pub topic_count: usize,
    pub root_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LabelingStats {
    pub notes_processed: usize,
    pub assignments_created: usize,
    pub topics_matched: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GapKind {
    #[default]
    Missing,
    Undercovered,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TopicGap {
    pub topic_id: i64,
    pub path: String,
    pub label: String,
    pub description: Option<String>,
    pub gap_type: GapKind,
    pub note_count: i64,
    pub threshold: i64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nearest_notes: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct WeakNote {
    pub note_id: NoteId,
    pub topic_path: String,
    pub confidence: f64,
    pub lapses: i32,
    pub fail_rate: Option<f64>,
    pub normalized_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DuplicateDetail {
    pub note_id: NoteId,
    pub similarity: f64,
    pub text: String,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DuplicateCluster {
    pub representative_id: NoteId,
    pub representative_text: String,
    pub duplicates: Vec<DuplicateDetail>,
    pub deck_names: Vec<String>,
    pub tags: Vec<String>,
}

impl DuplicateCluster {
    pub fn size(&self) -> usize {
        1 + self.duplicates.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DuplicateStats {
    pub notes_scanned: usize,
    pub clusters_found: usize,
    pub total_duplicates: usize,
    pub avg_cluster_size: f64,
}

#[cfg(test)]
mod tests {
    use super::{
        DuplicateCluster, DuplicateDetail, DuplicateStats, GapKind, LabelingStats, NoteId,
        TaxonomyLoadSummary, TopicCoverage, TopicGap, WeakNote,
    };

    #[test]
    fn duplicate_cluster_size_counts_representative() {
        let cluster = DuplicateCluster {
            duplicates: vec![DuplicateDetail::default(), DuplicateDetail::default()],
            ..Default::default()
        };

        assert_eq!(cluster.size(), 3);
    }

    #[test]
    fn analytics_contracts_round_trip_through_json() {
        let payload = (
            TaxonomyLoadSummary {
                topic_count: 12,
                root_count: 3,
            },
            LabelingStats {
                notes_processed: 5,
                assignments_created: 7,
                topics_matched: 4,
            },
            TopicCoverage {
                topic_id: 1,
                path: "rust/ownership".to_string(),
                label: "Ownership".to_string(),
                note_count: 3,
                subtree_count: 2,
                child_count: 1,
                covered_children: 1,
                mature_count: 2,
                avg_confidence: 0.8,
                weak_notes: 1,
                avg_lapses: 0.4,
            },
            TopicGap {
                topic_id: 2,
                path: "rust/borrowing".to_string(),
                label: "Borrowing".to_string(),
                description: Some("desc".to_string()),
                gap_type: GapKind::Missing,
                note_count: 0,
                threshold: 1,
                nearest_notes: Vec::new(),
            },
            WeakNote {
                note_id: NoteId(10),
                topic_path: "rust/ownership".to_string(),
                confidence: 0.55,
                lapses: 4,
                fail_rate: Some(0.25),
                normalized_text: "preview".to_string(),
            },
            DuplicateStats {
                notes_scanned: 20,
                clusters_found: 2,
                total_duplicates: 4,
                avg_cluster_size: 3.0,
            },
        );

        let json = serde_json::to_string(&payload).expect("serialize analytics payload");
        let decoded: (
            TaxonomyLoadSummary,
            LabelingStats,
            TopicCoverage,
            TopicGap,
            WeakNote,
            DuplicateStats,
        ) = serde_json::from_str(&json).expect("deserialize analytics payload");

        assert_eq!(decoded, payload);
    }
}

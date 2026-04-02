use std::collections::HashMap;

use crate::models::WorkItem;

/// Maximum items per kind-based cluster batch before starting a new one.
const KIND_BATCH_SIZE: usize = 10;

/// Groups work items into clusters and assigns `cluster_id`.
///
/// Clustering rules (applied in order):
/// 1. Items with the same slug share a cluster: `cluster_id = slug`.
/// 2. Items with no slug but the same `IssueKind` variant are batched into
///    numbered clusters: `cluster_id = "{kind_tag}:{batch_n}"`.
pub struct ClusterBuilder;

impl ClusterBuilder {
    /// Assign `cluster_id` to each item in-place. Returns the mutated vec.
    pub fn assign(items: Vec<WorkItem>) -> Vec<WorkItem> {
        // Counters for kind-based batching: variant tag -> (current_batch, count_in_batch)
        let mut kind_batch: HashMap<String, (usize, usize)> = HashMap::new();

        items
            .into_iter()
            .map(|mut item| {
                if let Some(slug) = &item.slug {
                    // Slug-based cluster: all issues for the same card go together.
                    item.cluster_id = Some(slug.clone());
                } else {
                    // Kind-based cluster: group by IssueKind variant name.
                    let kind_tag = issue_kind_tag(&item.issue_kind);
                    let entry = kind_batch.entry(kind_tag.clone()).or_insert((0, 0));
                    if entry.1 >= KIND_BATCH_SIZE {
                        // Start a new batch
                        entry.0 += 1;
                        entry.1 = 0;
                    }
                    let batch_n = entry.0;
                    entry.1 += 1;
                    item.cluster_id = Some(format!("{kind_tag}:{batch_n}"));
                }
                item
            })
            .collect()
    }
}

/// Returns a stable string tag for the IssueKind variant (no payload).
fn issue_kind_tag(kind: &crate::models::IssueKind) -> String {
    use crate::models::IssueKind::*;
    match kind {
        LowQuality { .. } => "low_quality",
        ValidationError { .. } => "validation_error",
        Duplicate { .. } => "duplicate",
        SemanticOverlap { .. } => "semantic_overlap",
        SplitCandidate { .. } => "split_candidate",
        StaleContent => "stale_content",
        DeadSkill => "dead_skill",
        MissingTags => "missing_tags",
        UncoveredTopic { .. } => "uncovered_topic",
        MissingLanguage { .. } => "missing_language",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{IssueKind, ItemStatus, LoopKind, Tier};
    use chrono::Utc;
    use uuid::Uuid;

    fn make_item(slug: Option<&str>, issue_kind: IssueKind) -> WorkItem {
        WorkItem {
            id: Uuid::new_v4().to_string(),
            loop_kind: LoopKind::Audit,
            issue_kind,
            tier: Tier::AutoFix,
            status: ItemStatus::Open,
            slug: slug.map(|s| s.to_string()),
            source_path: "notes/test.md".into(),
            summary: "test".into(),
            detail: None,
            first_seen: Utc::now(),
            resolved_at: None,
            attestation: None,
            scan_number: 1,
            cluster_id: None,
            confidence: None,
        }
    }

    #[test]
    fn slug_based_cluster() {
        let items = vec![
            make_item(Some("card-a"), IssueKind::MissingTags),
            make_item(Some("card-a"), IssueKind::DeadSkill),
            make_item(Some("card-b"), IssueKind::MissingTags),
        ];

        let result = ClusterBuilder::assign(items);

        assert_eq!(result[0].cluster_id.as_deref(), Some("card-a"));
        assert_eq!(result[1].cluster_id.as_deref(), Some("card-a"));
        assert_eq!(result[2].cluster_id.as_deref(), Some("card-b"));
    }

    #[test]
    fn kind_based_cluster_batch() {
        // Items without slugs of the same kind should share a cluster until batch is full.
        let items: Vec<WorkItem> = (0..KIND_BATCH_SIZE + 1)
            .map(|_| {
                make_item(
                    None,
                    IssueKind::UncoveredTopic {
                        topic: "rust".into(),
                    },
                )
            })
            .collect();

        let result = ClusterBuilder::assign(items);

        // First KIND_BATCH_SIZE items in batch 0
        for item in &result[..KIND_BATCH_SIZE] {
            assert_eq!(item.cluster_id.as_deref(), Some("uncovered_topic:0"));
        }
        // The overflow item starts batch 1
        assert_eq!(
            result[KIND_BATCH_SIZE].cluster_id.as_deref(),
            Some("uncovered_topic:1")
        );
    }

    #[test]
    fn different_kinds_get_different_clusters() {
        let items = vec![
            make_item(None, IssueKind::MissingTags),
            make_item(None, IssueKind::UncoveredTopic { topic: "go".into() }),
        ];

        let result = ClusterBuilder::assign(items);

        assert_eq!(result[0].cluster_id.as_deref(), Some("missing_tags:0"));
        assert_eq!(result[1].cluster_id.as_deref(), Some("uncovered_topic:0"));
    }
}

use crate::error::CardloopError;
use crate::models::{LoopKind, WorkItem};
use crate::store::CardloopStore;

/// Builds a prioritized execution queue from the store.
///
/// Priority order: Tier 1 (AutoFix) -> Tier 2 (QuickFix) -> Tier 3 (Rework) -> Tier 4 (Delete).
/// Within the same tier, oldest items first.
pub struct QueueBuilder;

impl QueueBuilder {
    /// Build a queue of the next `count` items to work on.
    ///
    /// The store's `query_open` already returns items sorted by tier then first_seen,
    /// so we over-fetch slightly to allow for any future filtering, then truncate.
    pub fn build(
        store: &CardloopStore,
        loop_kind: Option<LoopKind>,
        count: usize,
    ) -> Result<Vec<WorkItem>, CardloopError> {
        let mut items = store.query_open(loop_kind, count * 3)?;

        // Store already sorts by tier ASC, first_seen ASC.
        // Re-sort here to guarantee ordering even if store implementation changes.
        items.sort_by(|a, b| a.tier.cmp(&b.tier).then(a.first_seen.cmp(&b.first_seen)));

        items.truncate(count);
        Ok(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{IssueKind, ItemStatus, Tier};
    use chrono::{Duration, Utc};
    use uuid::Uuid;

    fn make_item(tier: Tier, loop_kind: LoopKind, age_hours: i64) -> WorkItem {
        WorkItem {
            id: Uuid::new_v4().to_string(),
            loop_kind,
            issue_kind: IssueKind::MissingTags,
            tier,
            status: ItemStatus::Open,
            slug: Some("slug".into()),
            source_path: "notes/test.md".into(),
            summary: format!("tier={tier:?} age={age_hours}h"),
            detail: None,
            first_seen: Utc::now() - Duration::hours(age_hours),
            resolved_at: None,
            attestation: None,
            scan_number: 1,
        }
    }

    #[test]
    fn queue_orders_by_tier_then_age() {
        let store = CardloopStore::open(":memory:").unwrap();

        let rework_old = make_item(Tier::Rework, LoopKind::Audit, 48);
        let autofix_new = make_item(Tier::AutoFix, LoopKind::Audit, 1);
        let autofix_old = make_item(Tier::AutoFix, LoopKind::Audit, 24);
        let quickfix = make_item(Tier::QuickFix, LoopKind::Audit, 12);

        store
            .upsert_items(&[
                rework_old.clone(),
                autofix_new.clone(),
                autofix_old.clone(),
                quickfix.clone(),
            ])
            .unwrap();

        let queue = QueueBuilder::build(&store, None, 10).unwrap();
        assert_eq!(queue.len(), 4);

        // AutoFix first, oldest before newest
        assert_eq!(queue[0].id, autofix_old.id);
        assert_eq!(queue[1].id, autofix_new.id);
        // Then QuickFix
        assert_eq!(queue[2].id, quickfix.id);
        // Then Rework
        assert_eq!(queue[3].id, rework_old.id);
    }

    #[test]
    fn queue_respects_count_limit() {
        let store = CardloopStore::open(":memory:").unwrap();

        let items: Vec<WorkItem> = (0..5)
            .map(|i| make_item(Tier::QuickFix, LoopKind::Audit, i))
            .collect();
        store.upsert_items(&items).unwrap();

        let queue = QueueBuilder::build(&store, None, 2).unwrap();
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn queue_filters_by_loop_kind() {
        let store = CardloopStore::open(":memory:").unwrap();

        let audit = make_item(Tier::AutoFix, LoopKind::Audit, 1);
        let gen_item = make_item(Tier::AutoFix, LoopKind::Generation, 1);
        store.upsert_items(&[audit, gen_item]).unwrap();

        let audit_queue = QueueBuilder::build(&store, Some(LoopKind::Audit), 10).unwrap();
        assert_eq!(audit_queue.len(), 1);
        assert_eq!(audit_queue[0].loop_kind, LoopKind::Audit);

        let gen_queue = QueueBuilder::build(&store, Some(LoopKind::Generation), 10).unwrap();
        assert_eq!(gen_queue.len(), 1);
    }

    #[test]
    fn queue_empty_store() {
        let store = CardloopStore::open(":memory:").unwrap();
        let queue = QueueBuilder::build(&store, None, 10).unwrap();
        assert!(queue.is_empty());
    }
}

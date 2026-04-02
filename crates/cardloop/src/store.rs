use std::collections::HashSet;
use std::path::Path;

use chrono::{DateTime, Utc};
use rusqlite::params;

use crate::error::CardloopError;
use crate::models::{IssueKind, ItemStatus, LoopKind, ScoreSummary, Tier, WorkItem};

/// Schema version for migration tracking.
pub const SCHEMA_VERSION: u32 = 2;

/// Column list for work_items queries (must match row_to_work_item field order).
const ITEM_COLUMNS: &str = "id, loop_kind, issue_kind_json, tier, status, slug, source_path, \
     summary, detail, first_seen, resolved_at, attestation, scan_number, cluster_id, confidence";

/// SQLite-backed persistent store for cardloop work items.
pub struct CardloopStore {
    conn: rusqlite::Connection,
}

fn parse_datetime(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s).ok().map(|dt| dt.to_utc())
}

fn format_datetime(dt: &DateTime<Utc>) -> String {
    dt.to_rfc3339()
}

fn row_to_work_item(row: &rusqlite::Row) -> rusqlite::Result<WorkItem> {
    let loop_kind_str: String = row.get(1)?;
    let issue_kind_json: String = row.get(2)?;
    let tier_i32: i32 = row.get(3)?;
    let status_str: String = row.get(4)?;
    let first_seen_str: String = row.get(9)?;
    let resolved_at_str: Option<String> = row.get(10)?;

    let loop_kind = loop_kind_str.parse::<LoopKind>().unwrap_or(LoopKind::Audit);
    let issue_kind: IssueKind =
        serde_json::from_str(&issue_kind_json).unwrap_or(IssueKind::StaleContent);
    let tier = Tier::from_i32(tier_i32).unwrap_or(Tier::Rework);
    let status = status_str.parse::<ItemStatus>().unwrap_or(ItemStatus::Open);

    Ok(WorkItem {
        id: row.get(0)?,
        loop_kind,
        issue_kind,
        tier,
        status,
        slug: row.get(5)?,
        source_path: row.get(6)?,
        summary: row.get(7)?,
        detail: row.get(8)?,
        first_seen: parse_datetime(&first_seen_str).unwrap_or_else(Utc::now),
        resolved_at: resolved_at_str.as_deref().and_then(parse_datetime),
        attestation: row.get(11)?,
        scan_number: row.get::<_, i64>(12)? as u32,
        cluster_id: row.get(13)?,
        confidence: row.get(14)?,
    })
}

impl CardloopStore {
    /// Open or create a store. Use ":memory:" for tests.
    /// Creates the `.cardloop/` directory if `db_path` is a file path.
    pub fn open(db_path: &str) -> Result<Self, CardloopError> {
        if db_path != ":memory:" {
            if let Some(parent) = Path::new(db_path).parent() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let conn = if db_path == ":memory:" {
            rusqlite::Connection::open_in_memory()?
        } else {
            rusqlite::Connection::open(db_path)?
        };

        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;

        let store = Self { conn };
        store.run_migrations()?;
        Ok(store)
    }

    fn run_migrations(&self) -> Result<(), CardloopError> {
        let version = self.get_schema_version()?;
        match version {
            0 => self.create_schema_v2()?,
            1 => self.migrate_v1_to_v2()?,
            _ => {} // Already at current version
        }
        Ok(())
    }

    fn get_schema_version(&self) -> Result<u32, CardloopError> {
        let exists: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version')",
            [],
            |row| row.get(0),
        )?;

        if !exists {
            return Ok(0);
        }

        let version: u32 = self.conn.query_row(
            "SELECT version FROM schema_version WHERE id = 1",
            [],
            |row| row.get(0),
        )?;

        Ok(version)
    }

    fn create_schema_v2(&self) -> Result<(), CardloopError> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS work_items (
                id TEXT PRIMARY KEY,
                loop_kind TEXT NOT NULL,
                issue_kind_json TEXT NOT NULL,
                tier INTEGER NOT NULL,
                status TEXT NOT NULL,
                slug TEXT,
                source_path TEXT NOT NULL,
                summary TEXT NOT NULL,
                detail TEXT,
                first_seen TEXT NOT NULL,
                resolved_at TEXT,
                attestation TEXT,
                scan_number INTEGER NOT NULL,
                cluster_id TEXT,
                confidence REAL
            );

            CREATE TABLE IF NOT EXISTS session_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_work_items_status
                ON work_items(status);
            CREATE INDEX IF NOT EXISTS idx_work_items_loop_kind
                ON work_items(loop_kind, status);
            CREATE INDEX IF NOT EXISTS idx_work_items_tier
                ON work_items(tier, status);
            CREATE INDEX IF NOT EXISTS idx_work_items_slug
                ON work_items(slug);
            CREATE INDEX IF NOT EXISTS idx_work_items_cluster
                ON work_items(cluster_id);

            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL
            );

            INSERT OR REPLACE INTO schema_version (id, version) VALUES (1, 2);
            ",
        )?;

        // Initialize session metadata.
        self.conn.execute(
            "INSERT OR IGNORE INTO session_meta (key, value) VALUES ('scan_count', '0')",
            [],
        )?;
        self.conn.execute(
            "INSERT OR IGNORE INTO session_meta (key, value) VALUES ('created', ?1)",
            params![format_datetime(&Utc::now())],
        )?;

        Ok(())
    }

    fn migrate_v1_to_v2(&self) -> Result<(), CardloopError> {
        self.conn.execute_batch(
            "ALTER TABLE work_items ADD COLUMN cluster_id TEXT;
             ALTER TABLE work_items ADD COLUMN confidence REAL;
             UPDATE schema_version SET version = 2 WHERE id = 1;",
        )?;
        Ok(())
    }

    // --- Work Item CRUD ---

    /// Insert or update work items. On conflict (same id), updates status-independent fields.
    pub fn upsert_items(&self, items: &[WorkItem]) -> Result<(), CardloopError> {
        let tx = self.conn.unchecked_transaction()?;

        {
            let mut stmt = tx.prepare(
                "INSERT INTO work_items (id, loop_kind, issue_kind_json, tier, status,
                    slug, source_path, summary, detail, first_seen, resolved_at,
                    attestation, scan_number, cluster_id, confidence)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
                 ON CONFLICT(id) DO UPDATE SET
                    issue_kind_json = excluded.issue_kind_json,
                    tier = excluded.tier,
                    summary = excluded.summary,
                    detail = excluded.detail,
                    scan_number = excluded.scan_number,
                    cluster_id = excluded.cluster_id,
                    confidence = excluded.confidence",
            )?;

            for item in items {
                let issue_json =
                    serde_json::to_string(&item.issue_kind).map_err(CardloopError::Json)?;
                let first_seen_str = format_datetime(&item.first_seen);
                let resolved_at_str = item.resolved_at.map(|dt| format_datetime(&dt));

                stmt.execute(params![
                    item.id,
                    item.loop_kind.as_str(),
                    issue_json,
                    item.tier.as_i32(),
                    item.status.as_str(),
                    item.slug,
                    item.source_path,
                    item.summary,
                    item.detail,
                    first_seen_str,
                    resolved_at_str,
                    item.attestation,
                    item.scan_number as i64,
                    item.cluster_id,
                    item.confidence,
                ])?;
            }
        }

        tx.commit()?;
        Ok(())
    }

    /// Get a single work item by id.
    pub fn get_item(&self, id: &str) -> Result<Option<WorkItem>, CardloopError> {
        let sql = format!("SELECT {ITEM_COLUMNS} FROM work_items WHERE id = ?1");
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query_map(params![id], row_to_work_item)?;
        Ok(rows.next().transpose()?)
    }

    /// Get a work item by id prefix match (for CLI convenience).
    pub fn get_item_by_prefix(&self, prefix: &str) -> Result<Option<WorkItem>, CardloopError> {
        let sql = format!("SELECT {ITEM_COLUMNS} FROM work_items WHERE id LIKE ?1 LIMIT 2");
        let pattern = format!("{prefix}%");
        let mut stmt = self.conn.prepare(&sql)?;
        let items: Vec<WorkItem> = stmt
            .query_map(params![pattern], row_to_work_item)?
            .collect::<Result<Vec<_>, _>>()?;

        match items.len() {
            0 => Ok(None),
            1 => Ok(Some(items.into_iter().next().expect("checked len"))),
            _ => Ok(None), // Ambiguous prefix
        }
    }

    /// Transition a work item to a new status with optional attestation.
    pub fn transition(
        &self,
        id: &str,
        new_status: ItemStatus,
        attestation: Option<&str>,
    ) -> Result<WorkItem, CardloopError> {
        let item = self
            .get_item(id)?
            .ok_or_else(|| CardloopError::NotFound(id.to_string()))?;

        // Validate transition
        let valid = matches!(
            (item.status, new_status),
            (ItemStatus::Open, ItemStatus::InProgress)
                | (ItemStatus::Open, ItemStatus::Fixed)
                | (ItemStatus::Open, ItemStatus::Skipped)
                | (ItemStatus::Open, ItemStatus::WontFix)
                | (ItemStatus::InProgress, ItemStatus::Fixed)
                | (ItemStatus::InProgress, ItemStatus::Skipped)
                | (ItemStatus::InProgress, ItemStatus::WontFix)
                | (ItemStatus::InProgress, ItemStatus::Open)
                | (ItemStatus::Fixed, ItemStatus::Open) // auto-reopen after verification gate
        );

        if !valid {
            return Err(CardloopError::InvalidTransition {
                id: id.to_string(),
                from: item.status.as_str().to_string(),
                to: new_status.as_str().to_string(),
            });
        }

        let resolved_at = if new_status.is_resolved() {
            Some(format_datetime(&Utc::now()))
        } else {
            None
        };

        self.conn.execute(
            "UPDATE work_items SET status = ?1, attestation = ?2, resolved_at = ?3 WHERE id = ?4",
            params![new_status.as_str(), attestation, resolved_at, id],
        )?;

        self.get_item(id)?
            .ok_or_else(|| CardloopError::NotFound(id.to_string()))
    }

    // --- Queries ---

    /// Query open (or in_progress) items, optionally filtered by loop kind.
    pub fn query_open(
        &self,
        loop_kind: Option<LoopKind>,
        limit: usize,
    ) -> Result<Vec<WorkItem>, CardloopError> {
        let (sql, param_values): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match loop_kind {
            Some(lk) => (
                format!(
                    "SELECT {ITEM_COLUMNS} FROM work_items \
                     WHERE status IN ('open', 'in_progress') AND loop_kind = ?1 \
                     ORDER BY tier ASC, first_seen ASC LIMIT ?2"
                ),
                vec![
                    Box::new(lk.as_str().to_string()) as Box<dyn rusqlite::types::ToSql>,
                    Box::new(limit as i64),
                ],
            ),
            None => (
                format!(
                    "SELECT {ITEM_COLUMNS} FROM work_items \
                     WHERE status IN ('open', 'in_progress') \
                     ORDER BY tier ASC, first_seen ASC LIMIT ?1"
                ),
                vec![Box::new(limit as i64) as Box<dyn rusqlite::types::ToSql>],
            ),
        };

        let param_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(param_refs.as_slice(), row_to_work_item)?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Query all work items for a given card slug.
    pub fn query_by_slug(&self, slug: &str) -> Result<Vec<WorkItem>, CardloopError> {
        let sql = format!("SELECT {ITEM_COLUMNS} FROM work_items WHERE slug = ?1");
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![slug], row_to_work_item)?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    // --- Scoring ---

    /// Compute a score summary across all work items.
    pub fn compute_scores(&self) -> Result<ScoreSummary, CardloopError> {
        let open_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM work_items WHERE status IN ('open', 'in_progress')",
            [],
            |row| row.get(0),
        )?;

        let fixed_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM work_items WHERE status = 'fixed'",
            [],
            |row| row.get(0),
        )?;

        let skipped_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM work_items WHERE status IN ('skipped', 'wontfix')",
            [],
            |row| row.get(0),
        )?;

        // By tier (open items only)
        let mut by_tier = [0usize; 4];
        {
            let mut stmt = self.conn.prepare(
                "SELECT tier, COUNT(*) FROM work_items \
                 WHERE status IN ('open', 'in_progress') GROUP BY tier",
            )?;
            let rows =
                stmt.query_map([], |row| Ok((row.get::<_, i32>(0)?, row.get::<_, i64>(1)?)))?;
            for row in rows {
                let (tier, count) = row?;
                if (1..=4).contains(&tier) {
                    by_tier[(tier - 1) as usize] = count as usize;
                }
            }
        }

        // By loop kind (open items only)
        let gen_open: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM work_items \
             WHERE status IN ('open', 'in_progress') AND loop_kind = 'generation'",
            [],
            |row| row.get(0),
        )?;
        let audit_open: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM work_items \
             WHERE status IN ('open', 'in_progress') AND loop_kind = 'audit'",
            [],
            |row| row.get(0),
        )?;

        // Lenient score: fixed / (fixed + open). WontFix/Skipped excluded from denominator.
        let overall = if fixed_count + open_count > 0 {
            fixed_count as f64 / (fixed_count + open_count) as f64
        } else {
            1.0
        };

        // Strict score: fixed / total (wontfix/skipped count against you).
        let total = open_count + fixed_count + skipped_count;
        let strict_score = if total > 0 {
            fixed_count as f64 / total as f64
        } else {
            1.0
        };

        Ok(ScoreSummary {
            overall,
            strict_score,
            quality_avg: 0.0, // Populated by scanner, not from DB alone
            open_count: open_count as usize,
            fixed_count: fixed_count as usize,
            skipped_count: skipped_count as usize,
            by_tier,
            by_loop: (gen_open as usize, audit_open as usize),
            health_score: None, // Requires anki-reader FSRS integration
        })
    }

    // --- Session metadata ---

    /// Increment and return the new scan count.
    pub fn increment_scan_count(&self) -> Result<u32, CardloopError> {
        self.conn.execute(
            "UPDATE session_meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) \
             WHERE key = 'scan_count'",
            [],
        )?;

        let count: String = self.conn.query_row(
            "SELECT value FROM session_meta WHERE key = 'scan_count'",
            [],
            |row| row.get(0),
        )?;

        Ok(count.parse::<u32>().unwrap_or(0))
    }

    /// Get current scan count.
    pub fn scan_count(&self) -> Result<u32, CardloopError> {
        let count: String = self.conn.query_row(
            "SELECT value FROM session_meta WHERE key = 'scan_count'",
            [],
            |row| row.get(0),
        )?;
        Ok(count.parse::<u32>().unwrap_or(0))
    }

    // --- Reconciliation ---

    /// Auto-resolve open items whose IDs are not in `current_ids`.
    /// Returns the number of items auto-resolved.
    pub fn reconcile_stale(&self, current_ids: &HashSet<String>) -> Result<usize, CardloopError> {
        let sql = format!(
            "SELECT {ITEM_COLUMNS} FROM work_items WHERE status IN ('open', 'in_progress')"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let open_items: Vec<WorkItem> = stmt
            .query_map([], row_to_work_item)?
            .collect::<Result<Vec<_>, _>>()?;

        let mut resolved = 0;
        for item in &open_items {
            if !current_ids.contains(&item.id) {
                self.conn.execute(
                    "UPDATE work_items SET status = 'fixed', \
                     attestation = 'auto-resolved: no longer detected', \
                     resolved_at = ?1 WHERE id = ?2",
                    params![format_datetime(&Utc::now()), item.id],
                )?;
                resolved += 1;
            }
        }

        Ok(resolved)
    }

    /// Total count of all work items.
    pub fn total_count(&self) -> Result<usize, CardloopError> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM work_items", [], |row| row.get(0))?;
        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::IssueKind;
    use uuid::Uuid;

    fn make_item(tier: Tier, loop_kind: LoopKind) -> WorkItem {
        WorkItem {
            id: Uuid::new_v4().to_string(),
            loop_kind,
            issue_kind: IssueKind::MissingTags,
            tier,
            status: ItemStatus::Open,
            slug: Some("test-card-slug".into()),
            source_path: "notes/test.md".into(),
            summary: "Missing topic tags".into(),
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
    fn open_in_memory() {
        let store = CardloopStore::open(":memory:").unwrap();
        assert_eq!(store.scan_count().unwrap(), 0);
        assert_eq!(store.total_count().unwrap(), 0);
    }

    #[test]
    fn upsert_and_get() {
        let store = CardloopStore::open(":memory:").unwrap();
        let item = make_item(Tier::QuickFix, LoopKind::Audit);
        let id = item.id.clone();

        store.upsert_items(&[item]).unwrap();
        assert_eq!(store.total_count().unwrap(), 1);

        let fetched = store.get_item(&id).unwrap().unwrap();
        assert_eq!(fetched.id, id);
        assert_eq!(fetched.tier, Tier::QuickFix);
        assert_eq!(fetched.status, ItemStatus::Open);
    }

    #[test]
    fn upsert_updates_existing() {
        let store = CardloopStore::open(":memory:").unwrap();
        let mut item = make_item(Tier::QuickFix, LoopKind::Audit);
        let id = item.id.clone();

        store.upsert_items(&[item.clone()]).unwrap();

        // Update summary and tier via upsert
        item.summary = "Updated summary".into();
        item.tier = Tier::Rework;
        store.upsert_items(&[item]).unwrap();

        assert_eq!(store.total_count().unwrap(), 1);
        let fetched = store.get_item(&id).unwrap().unwrap();
        assert_eq!(fetched.summary, "Updated summary");
        assert_eq!(fetched.tier, Tier::Rework);
        // Status should NOT be overwritten by upsert
        assert_eq!(fetched.status, ItemStatus::Open);
    }

    #[test]
    fn transition_valid() {
        let store = CardloopStore::open(":memory:").unwrap();
        let item = make_item(Tier::AutoFix, LoopKind::Audit);
        let id = item.id.clone();
        store.upsert_items(&[item]).unwrap();

        let updated = store
            .transition(&id, ItemStatus::Fixed, Some("added tags"))
            .unwrap();
        assert_eq!(updated.status, ItemStatus::Fixed);
        assert_eq!(updated.attestation.as_deref(), Some("added tags"));
        assert!(updated.resolved_at.is_some());
    }

    #[test]
    fn transition_invalid() {
        let store = CardloopStore::open(":memory:").unwrap();
        let item = make_item(Tier::AutoFix, LoopKind::Audit);
        let id = item.id.clone();
        store.upsert_items(&[item]).unwrap();

        // Skip it first
        store.transition(&id, ItemStatus::Skipped, None).unwrap();

        // Skipped -> InProgress should fail (not a valid transition)
        let result = store.transition(&id, ItemStatus::InProgress, None);
        assert!(result.is_err());
    }

    #[test]
    fn transition_fixed_to_open_reopen() {
        let store = CardloopStore::open(":memory:").unwrap();
        let item = make_item(Tier::AutoFix, LoopKind::Audit);
        let id = item.id.clone();
        store.upsert_items(&[item]).unwrap();

        store
            .transition(&id, ItemStatus::Fixed, Some("done"))
            .unwrap();

        // Fixed -> Open is valid (auto-reopen after verification gate)
        let reopened = store
            .transition(&id, ItemStatus::Open, Some("auto-reopened"))
            .unwrap();
        assert_eq!(reopened.status, ItemStatus::Open);
    }

    #[test]
    fn query_open_by_loop_kind() {
        let store = CardloopStore::open(":memory:").unwrap();

        let audit1 = make_item(Tier::QuickFix, LoopKind::Audit);
        let audit2 = make_item(Tier::AutoFix, LoopKind::Audit);
        let gen1 = make_item(Tier::Rework, LoopKind::Generation);

        store.upsert_items(&[audit1, audit2, gen1]).unwrap();

        let audit_items = store.query_open(Some(LoopKind::Audit), 10).unwrap();
        assert_eq!(audit_items.len(), 2);
        // Should be ordered by tier: AutoFix first
        assert_eq!(audit_items[0].tier, Tier::AutoFix);

        let gen_items = store.query_open(Some(LoopKind::Generation), 10).unwrap();
        assert_eq!(gen_items.len(), 1);

        let all_items = store.query_open(None, 10).unwrap();
        assert_eq!(all_items.len(), 3);
    }

    #[test]
    fn query_by_slug() {
        let store = CardloopStore::open(":memory:").unwrap();
        let item = make_item(Tier::QuickFix, LoopKind::Audit);
        store.upsert_items(&[item]).unwrap();

        let results = store.query_by_slug("test-card-slug").unwrap();
        assert_eq!(results.len(), 1);

        let empty = store.query_by_slug("nonexistent").unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn compute_scores() {
        let store = CardloopStore::open(":memory:").unwrap();

        let item1 = make_item(Tier::AutoFix, LoopKind::Audit);
        let item2 = make_item(Tier::QuickFix, LoopKind::Generation);
        let item3 = make_item(Tier::Rework, LoopKind::Audit);
        let id3 = item3.id.clone();

        store.upsert_items(&[item1, item2, item3]).unwrap();

        // Fix one item
        store
            .transition(&id3, ItemStatus::Fixed, Some("done"))
            .unwrap();

        let scores = store.compute_scores().unwrap();
        assert_eq!(scores.open_count, 2);
        assert_eq!(scores.fixed_count, 1);
        assert_eq!(scores.skipped_count, 0);
        assert_eq!(scores.by_loop, (1, 1)); // gen=1 open, audit=1 open
        // overall is lenient: fixed/(fixed+open) = 1/3
        assert!((scores.overall - 1.0 / 3.0).abs() < 0.01);
        // strict_score: fixed/total = 1/3 (same here since no skipped)
        assert!((scores.strict_score - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn increment_scan_count() {
        let store = CardloopStore::open(":memory:").unwrap();
        assert_eq!(store.scan_count().unwrap(), 0);
        assert_eq!(store.increment_scan_count().unwrap(), 1);
        assert_eq!(store.increment_scan_count().unwrap(), 2);
        assert_eq!(store.scan_count().unwrap(), 2);
    }

    #[test]
    fn reconcile_stale_items() {
        let store = CardloopStore::open(":memory:").unwrap();

        let item1 = make_item(Tier::AutoFix, LoopKind::Audit);
        let item2 = make_item(Tier::QuickFix, LoopKind::Audit);
        let id1 = item1.id.clone();
        let id2 = item2.id.clone();

        store.upsert_items(&[item1, item2]).unwrap();

        // Only item1 is still current
        let mut current = HashSet::new();
        current.insert(id1.clone());

        let resolved = store.reconcile_stale(&current).unwrap();
        assert_eq!(resolved, 1);

        // item2 should be auto-resolved
        let item2_after = store.get_item(&id2).unwrap().unwrap();
        assert_eq!(item2_after.status, ItemStatus::Fixed);
        assert_eq!(
            item2_after.attestation.as_deref(),
            Some("auto-resolved: no longer detected")
        );

        // item1 should still be open
        let item1_after = store.get_item(&id1).unwrap().unwrap();
        assert_eq!(item1_after.status, ItemStatus::Open);
    }

    #[test]
    fn get_item_by_prefix() {
        let store = CardloopStore::open(":memory:").unwrap();
        let item = make_item(Tier::AutoFix, LoopKind::Audit);
        let id = item.id.clone();
        let prefix = &id[..8];
        store.upsert_items(&[item]).unwrap();

        let found = store.get_item_by_prefix(prefix).unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, id);
    }

    #[test]
    fn persistent_store() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("state.db");
        let db_str = db_path.to_str().unwrap();

        // Write
        {
            let store = CardloopStore::open(db_str).unwrap();
            let item = make_item(Tier::AutoFix, LoopKind::Audit);
            store.upsert_items(&[item]).unwrap();
            store.increment_scan_count().unwrap();
        }

        // Re-open and verify
        {
            let store = CardloopStore::open(db_str).unwrap();
            assert_eq!(store.total_count().unwrap(), 1);
            assert_eq!(store.scan_count().unwrap(), 1);
        }
    }
}

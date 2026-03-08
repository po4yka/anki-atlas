use std::collections::HashSet;

use anki_sync::{CardRecovery, CardState, CardTransaction, StateDB};
use tempfile::TempDir;

fn make_state(slug: &str, synced_at: f64) -> CardState {
    CardState {
        slug: slug.to_string(),
        content_hash: "hash".to_string(),
        anki_guid: None,
        note_type: "Basic".to_string(),
        source_path: "notes/test.md".to_string(),
        synced_at,
    }
}

// --- CardTransaction tests ---

#[test]
fn transaction_new_creates_empty() {
    let txn = CardTransaction::new();
    drop(txn); // Should not panic
}

#[test]
fn transaction_commit_prevents_rollback() {
    let mut txn = CardTransaction::new();
    txn.add_rollback("create", "card-1");
    txn.add_rollback("create", "card-2");
    txn.commit();

    let actions = txn.rollback();
    assert!(actions.is_empty());
}

#[test]
fn transaction_rollback_returns_actions_in_reverse() {
    let mut txn = CardTransaction::new();
    txn.add_rollback("create", "card-1");
    txn.add_rollback("update", "card-2");
    txn.add_rollback("delete", "card-3");

    let actions = txn.rollback();
    assert_eq!(actions.len(), 3);
    assert_eq!(actions[0].target_id, "card-3");
    assert_eq!(actions[0].action_type, "delete");
    assert_eq!(actions[1].target_id, "card-2");
    assert_eq!(actions[2].target_id, "card-1");
}

#[test]
fn transaction_rollback_after_rollback_is_empty() {
    let mut txn = CardTransaction::new();
    txn.add_rollback("create", "card-1");

    let first = txn.rollback();
    assert_eq!(first.len(), 1);

    let second = txn.rollback();
    assert!(second.is_empty());
}

// --- CardRecovery tests ---

#[test]
fn find_orphaned_computes_set_differences() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();
    let recovery = CardRecovery::new(&db);

    let db_slugs: HashSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    let anki_slugs: HashSet<String> = ["b", "c", "d"].iter().map(|s| s.to_string()).collect();

    let (in_db_not_anki, in_anki_not_db) = recovery.find_orphaned(&db_slugs, &anki_slugs);

    assert_eq!(in_db_not_anki, HashSet::from(["a".to_string()]));
    assert_eq!(in_anki_not_db, HashSet::from(["d".to_string()]));
}

#[test]
fn find_orphaned_empty_sets() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();
    let recovery = CardRecovery::new(&db);

    let empty: HashSet<String> = HashSet::new();
    let (in_db, in_anki) = recovery.find_orphaned(&empty, &empty);

    assert!(in_db.is_empty());
    assert!(in_anki.is_empty());
}

#[test]
fn find_orphaned_identical_sets() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();
    let recovery = CardRecovery::new(&db);

    let slugs: HashSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
    let (in_db, in_anki) = recovery.find_orphaned(&slugs, &slugs);

    assert!(in_db.is_empty());
    assert!(in_anki.is_empty());
}

#[test]
fn find_stale_returns_states_older_than_cutoff() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    // 100 days ago
    let old_time = now - 100.0 * 86400.0;
    // 1 day ago
    let recent_time = now - 1.0 * 86400.0;

    db.upsert(&make_state("old-card", old_time));
    db.upsert(&make_state("recent-card", recent_time));

    let recovery = CardRecovery::new(&db);
    let stale = recovery.find_stale(30); // 30 days max age

    assert_eq!(stale.len(), 1);
    assert_eq!(stale[0].slug, "old-card");
}

#[test]
fn find_stale_excludes_never_synced() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    // synced_at = 0 means never synced
    db.upsert(&make_state("never-synced", 0.0));
    // Very old but synced
    db.upsert(&make_state("old-synced", now - 365.0 * 86400.0));

    let recovery = CardRecovery::new(&db);
    let stale = recovery.find_stale(30);

    assert_eq!(stale.len(), 1);
    assert_eq!(stale[0].slug, "old-synced");
}

#[test]
fn find_stale_returns_empty_when_all_recent() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    db.upsert(&make_state("card-1", now - 1.0 * 86400.0));
    db.upsert(&make_state("card-2", now - 5.0 * 86400.0));

    let recovery = CardRecovery::new(&db);
    let stale = recovery.find_stale(30);

    assert!(stale.is_empty());
}

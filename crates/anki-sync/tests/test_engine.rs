use anki_sync::engine::{SyncEngine, SyncResult};
use anki_sync::progress::{ProgressTracker, SyncPhase};
use anki_sync::state::{CardState, StateDB};
use tempfile::TempDir;

/// Helper: create a StateDB in a temp directory.
fn make_state_db() -> (StateDB, TempDir) {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("state.db");
    let db = StateDB::open(&db_path).unwrap();
    (db, dir)
}

/// Helper: seed the state DB with N card states.
fn seed_states(db: &StateDB, count: usize) {
    for i in 0..count {
        db.upsert(&CardState {
            slug: format!("card-{i:03}"),
            content_hash: format!("hash-{i}"),
            anki_guid: Some(i as i64),
            note_type: "basic".to_string(),
            source_path: "notes/test.md".to_string(),
            synced_at: 1000.0 + i as f64,
        })
        .unwrap();
    }
}

// --- Construction ---

#[test]
fn new_with_explicit_progress() {
    let (db, _dir) = make_state_db();
    let progress = ProgressTracker::new(Some("test-session".to_string()));
    let engine = SyncEngine::new(db, Some(progress));

    // Engine should store the provided progress tracker
    let snap = engine.progress().snapshot();
    assert_eq!(snap.session_id, "test-session");
}

#[test]
fn new_with_none_creates_default_progress() {
    let (db, _dir) = make_state_db();
    let engine = SyncEngine::new(db, None);

    // Should have a generated session ID (UUID format)
    let snap = engine.progress().snapshot();
    assert!(!snap.session_id.is_empty());
    assert_eq!(snap.phase, SyncPhase::Initializing);
}

#[test]
fn state_db_accessor_returns_reference() {
    let (db, _dir) = make_state_db();
    db.upsert(&CardState {
        slug: "accessor-test".to_string(),
        content_hash: "h1".to_string(),
        anki_guid: None,
        note_type: "basic".to_string(),
        source_path: "test.md".to_string(),
        synced_at: 0.0,
    })
    .unwrap();

    let engine = SyncEngine::new(db, None);
    // Should be able to query via the accessor
    let state = engine.state_db().get("accessor-test").unwrap();
    assert!(state.is_some());
    assert_eq!(state.unwrap().slug, "accessor-test");
}

#[test]
fn progress_accessor_returns_reference() {
    let (db, _dir) = make_state_db();
    let progress = ProgressTracker::new(Some("ref-test".to_string()));
    let engine = SyncEngine::new(db, Some(progress));

    engine.progress().set_phase(SyncPhase::Scanning);
    let snap = engine.progress().snapshot();
    assert_eq!(snap.phase, SyncPhase::Scanning);
}

// --- Sync lifecycle: phase transitions ---

#[test]
fn sync_transitions_scanning_applying_completed() {
    let (db, _dir) = make_state_db();
    seed_states(&db, 3);

    let progress = ProgressTracker::new(Some("lifecycle".to_string()));
    let progress_clone = progress.clone();
    let mut engine = SyncEngine::new(db, Some(progress));

    let result = engine.sync(common::ExecutionMode::Execute).unwrap();

    // After successful sync, phase should be Completed
    let snap = progress_clone.snapshot();
    assert_eq!(snap.phase, SyncPhase::Completed);

    // Result should be returned
    assert!(result.duration_ms >= 0);
}

#[test]
fn sync_dry_run_skips_applying() {
    let (db, _dir) = make_state_db();
    seed_states(&db, 5);

    let progress = ProgressTracker::new(Some("dry-run".to_string()));
    let progress_clone = progress.clone();
    let mut engine = SyncEngine::new(db, Some(progress));

    let result = engine.sync(common::ExecutionMode::DryRun).unwrap();

    // Should complete successfully
    let snap = progress_clone.snapshot();
    assert_eq!(snap.phase, SyncPhase::Completed);

    // No changes should be applied in dry run
    assert_eq!(result.cards_created, 0);
    assert_eq!(result.cards_updated, 0);
    assert_eq!(result.cards_deleted, 0);
}

#[test]
fn sync_sets_total_from_existing_states() {
    let (db, _dir) = make_state_db();
    seed_states(&db, 7);

    let progress = ProgressTracker::new(Some("total-check".to_string()));
    let progress_clone = progress.clone();
    let mut engine = SyncEngine::new(db, Some(progress));

    engine.sync(true).unwrap();

    // During scanning, set_total should have been called with 7
    let snap = progress_clone.snapshot();
    assert_eq!(snap.total_notes, 7);
}

#[test]
fn sync_empty_state_db_succeeds() {
    let (db, _dir) = make_state_db();

    let progress = ProgressTracker::new(Some("empty-db".to_string()));
    let progress_clone = progress.clone();
    let mut engine = SyncEngine::new(db, Some(progress));

    let result = engine.sync(common::ExecutionMode::Execute).unwrap();

    let snap = progress_clone.snapshot();
    assert_eq!(snap.phase, SyncPhase::Completed);
    assert_eq!(snap.total_notes, 0);
    assert_eq!(result.cards_created, 0);
}

// --- SyncResult fields ---

#[test]
fn sync_result_has_non_negative_duration() {
    let (db, _dir) = make_state_db();
    let mut engine = SyncEngine::new(db, None);

    let result = engine.sync(common::ExecutionMode::Execute).unwrap();
    assert!(result.duration_ms >= 0);
}

#[test]
fn sync_result_errors_zero_on_success() {
    let (db, _dir) = make_state_db();
    let mut engine = SyncEngine::new(db, None);

    let result = engine.sync(common::ExecutionMode::Execute).unwrap();
    assert_eq!(result.errors, 0);
}

#[test]
fn sync_result_cards_skipped_zero_for_base_engine() {
    let (db, _dir) = make_state_db();
    seed_states(&db, 2);
    let mut engine = SyncEngine::new(db, None);

    let result = engine.sync(common::ExecutionMode::Execute).unwrap();
    assert_eq!(result.cards_skipped, 0);
}

// --- Multiple syncs ---

#[test]
fn sync_can_be_called_twice() {
    let (db, _dir) = make_state_db();
    seed_states(&db, 3);

    let mut engine = SyncEngine::new(db, None);

    let r1 = engine.sync(common::ExecutionMode::Execute).unwrap();
    let r2 = engine.sync(common::ExecutionMode::Execute).unwrap();

    // Both should succeed
    assert!(r1.duration_ms >= 0);
    assert!(r2.duration_ms >= 0);
}

// --- SyncResult Debug derive ---

#[test]
fn sync_result_implements_debug() {
    let result = SyncResult {
        cards_created: 1,
        cards_updated: 2,
        cards_deleted: 3,
        cards_skipped: 4,
        errors: 0,
        duration_ms: 100,
    };
    let debug = format!("{result:?}");
    assert!(debug.contains("cards_created"));
}

#[test]
fn sync_result_implements_clone() {
    let result = SyncResult {
        cards_created: 10,
        cards_updated: 20,
        cards_deleted: 0,
        cards_skipped: 0,
        errors: 0,
        duration_ms: 42,
    };
    let cloned = result.clone();
    assert_eq!(cloned.cards_created, 10);
    assert_eq!(cloned.duration_ms, 42);
}

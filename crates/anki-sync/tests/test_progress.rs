use anki_sync::{ProgressTracker, SyncPhase, VALID_STATS, progress::SyncStat};
use std::thread;

#[test]
fn new_creates_tracker_with_provided_session_id() {
    let tracker = ProgressTracker::new(Some("test-session".to_string()));
    let snap = tracker.snapshot();
    assert_eq!(snap.session_id, "test-session");
}

#[test]
fn new_creates_tracker_with_generated_session_id() {
    let tracker = ProgressTracker::new(None);
    let snap = tracker.snapshot();
    assert!(!snap.session_id.is_empty());
}

#[test]
fn initial_phase_is_initializing() {
    let tracker = ProgressTracker::new(None);
    let snap = tracker.snapshot();
    assert_eq!(snap.phase, SyncPhase::Initializing);
}

#[test]
fn set_phase_updates_phase() {
    let tracker = ProgressTracker::new(None);
    tracker.set_phase(SyncPhase::Scanning);

    let snap = tracker.snapshot();
    assert_eq!(snap.phase, SyncPhase::Scanning);
}

#[test]
fn set_phase_updates_updated_at() {
    let tracker = ProgressTracker::new(None);
    let snap1 = tracker.snapshot();
    let initial_updated = snap1.updated_at;

    // Small sleep to ensure timestamp differs
    std::thread::sleep(std::time::Duration::from_millis(10));

    tracker.set_phase(SyncPhase::Applying);
    let snap2 = tracker.snapshot();
    assert!(snap2.updated_at >= initial_updated);
}

#[test]
fn set_total_updates_total_notes() {
    let tracker = ProgressTracker::new(None);
    tracker.set_total(42);

    let snap = tracker.snapshot();
    assert_eq!(snap.total_notes, 42);
}

#[test]
fn increment_notes_processed() {
    let tracker = ProgressTracker::new(None);
    tracker.increment(SyncStat::NotesProcessed, 5);
    tracker.increment(SyncStat::NotesProcessed, 3);

    let snap = tracker.snapshot();
    assert_eq!(snap.notes_processed, 8);
}

#[test]
fn increment_cards_created() {
    let tracker = ProgressTracker::new(None);
    tracker.increment(SyncStat::CardsCreated, 10);

    let snap = tracker.snapshot();
    assert_eq!(snap.cards_created, 10);
}

#[test]
fn increment_cards_updated() {
    let tracker = ProgressTracker::new(None);
    tracker.increment(SyncStat::CardsUpdated, 7);

    let snap = tracker.snapshot();
    assert_eq!(snap.cards_updated, 7);
}

#[test]
fn increment_cards_deleted() {
    let tracker = ProgressTracker::new(None);
    tracker.increment(SyncStat::CardsDeleted, 2);

    let snap = tracker.snapshot();
    assert_eq!(snap.cards_deleted, 2);
}

#[test]
fn increment_errors() {
    let tracker = ProgressTracker::new(None);
    tracker.increment(SyncStat::Errors, 1);

    let snap = tracker.snapshot();
    assert_eq!(snap.errors, 1);
}

#[test]
fn complete_success_sets_completed_phase() {
    let tracker = ProgressTracker::new(None);
    tracker.complete(true);

    let snap = tracker.snapshot();
    assert_eq!(snap.phase, SyncPhase::Completed);
}

#[test]
fn complete_failure_sets_failed_phase() {
    let tracker = ProgressTracker::new(None);
    tracker.complete(false);

    let snap = tracker.snapshot();
    assert_eq!(snap.phase, SyncPhase::Failed);
}

#[test]
fn progress_pct_zero_when_total_is_zero() {
    let tracker = ProgressTracker::new(None);
    assert_eq!(tracker.progress_pct(), 0.0);
}

#[test]
fn progress_pct_correct_percentage() {
    let tracker = ProgressTracker::new(None);
    tracker.set_total(100);
    tracker.increment(SyncStat::NotesProcessed, 25);

    assert!((tracker.progress_pct() - 25.0).abs() < f64::EPSILON);
}

#[test]
fn progress_pct_caps_at_100() {
    let tracker = ProgressTracker::new(None);
    tracker.set_total(10);
    tracker.increment(SyncStat::NotesProcessed, 20);

    assert!(tracker.progress_pct() <= 100.0);
}

#[test]
fn thread_safety_concurrent_increments() {
    let tracker = ProgressTracker::new(None);
    let mut handles = vec![];

    for _ in 0..10 {
        let t = tracker.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                t.increment(SyncStat::NotesProcessed, 1);
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let snap = tracker.snapshot();
    assert_eq!(snap.notes_processed, 1000);
}

#[test]
fn clone_shares_state() {
    let tracker1 = ProgressTracker::new(Some("shared".to_string()));
    let tracker2 = tracker1.clone();

    tracker1.increment(SyncStat::CardsCreated, 5);
    let snap = tracker2.snapshot();
    assert_eq!(snap.cards_created, 5);
}

#[test]
fn sync_phase_as_str() {
    assert_eq!(SyncPhase::Initializing.as_str(), "initializing");
    assert_eq!(SyncPhase::Indexing.as_str(), "indexing");
    assert_eq!(SyncPhase::Scanning.as_str(), "scanning");
    assert_eq!(SyncPhase::Generating.as_str(), "generating");
    assert_eq!(SyncPhase::Applying.as_str(), "applying");
    assert_eq!(SyncPhase::Completed.as_str(), "completed");
    assert_eq!(SyncPhase::Failed.as_str(), "failed");
}

#[test]
fn valid_stats_contains_expected_names() {
    assert!(VALID_STATS.contains(&"notes_processed"));
    assert!(VALID_STATS.contains(&"cards_created"));
    assert!(VALID_STATS.contains(&"cards_updated"));
    assert!(VALID_STATS.contains(&"cards_deleted"));
    assert!(VALID_STATS.contains(&"errors"));
    assert_eq!(VALID_STATS.len(), 5);
}

#[test]
fn snapshot_returns_started_at() {
    let tracker = ProgressTracker::new(None);
    let snap = tracker.snapshot();
    assert!(snap.started_at > 0.0);
}

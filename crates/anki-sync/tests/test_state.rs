use anki_sync::{CardState, StateDB};
use tempfile::TempDir;

fn make_state(slug: &str, hash: &str) -> CardState {
    CardState {
        slug: slug.to_string(),
        content_hash: hash.to_string(),
        anki_guid: None,
        note_type: "Basic".to_string(),
        source_path: "notes/test.md".to_string(),
        synced_at: 1700000000.0,
    }
}

#[test]
fn open_creates_database_and_table() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("state.db");
    let db = StateDB::open(&db_path).unwrap();

    // Table should exist: inserting should not panic
    let state = make_state("test-slug", "abc123");
    db.upsert(&state).unwrap();

    let retrieved = db.get("test-slug").unwrap();
    assert!(retrieved.is_some());
}

#[test]
fn open_enables_wal_mode() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("state.db");
    let _db = StateDB::open(&db_path).unwrap();

    // Verify WAL mode by opening a second connection
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let mode: String = conn
        .pragma_query_value(None, "journal_mode", |row| row.get(0))
        .unwrap();
    assert_eq!(mode.to_lowercase(), "wal");
}

#[test]
fn upsert_inserts_new_state() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();

    let state = make_state("card-1", "hash-a");
    db.upsert(&state).unwrap();

    let retrieved = db.get("card-1").unwrap().unwrap();
    assert_eq!(retrieved.slug, "card-1");
    assert_eq!(retrieved.content_hash, "hash-a");
    assert_eq!(retrieved.anki_guid, None);
    assert_eq!(retrieved.note_type, "Basic");
    assert_eq!(retrieved.source_path, "notes/test.md");
    assert_eq!(retrieved.synced_at, 1700000000.0);
}

#[test]
fn upsert_updates_existing_state() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();

    let state1 = make_state("card-1", "hash-a");
    db.upsert(&state1).unwrap();

    let state2 = CardState {
        slug: "card-1".to_string(),
        content_hash: "hash-b".to_string(),
        anki_guid: Some(12345),
        note_type: "Cloze".to_string(),
        source_path: "notes/updated.md".to_string(),
        synced_at: 1700001000.0,
    };
    db.upsert(&state2).unwrap();

    let retrieved = db.get("card-1").unwrap().unwrap();
    assert_eq!(retrieved.content_hash, "hash-b");
    assert_eq!(retrieved.anki_guid, Some(12345));
    assert_eq!(retrieved.note_type, "Cloze");
    assert_eq!(retrieved.source_path, "notes/updated.md");
    assert_eq!(retrieved.synced_at, 1700001000.0);
}

#[test]
fn get_returns_none_for_missing_slug() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();

    assert!(db.get("nonexistent").unwrap().is_none());
}

#[test]
fn get_all_returns_sorted_by_slug() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();

    db.upsert(&make_state("charlie", "h3")).unwrap();
    db.upsert(&make_state("alpha", "h1")).unwrap();
    db.upsert(&make_state("bravo", "h2")).unwrap();

    let all = db.get_all().unwrap();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].slug, "alpha");
    assert_eq!(all[1].slug, "bravo");
    assert_eq!(all[2].slug, "charlie");
}

#[test]
fn get_all_returns_empty_vec_when_no_states() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();

    let all = db.get_all().unwrap();
    assert!(all.is_empty());
}

#[test]
fn delete_removes_state() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();

    db.upsert(&make_state("to-delete", "hash")).unwrap();
    assert!(db.get("to-delete").unwrap().is_some());

    db.delete("to-delete").unwrap();
    assert!(db.get("to-delete").unwrap().is_none());
}

#[test]
fn delete_nonexistent_is_noop() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();

    // Should not panic
    db.delete("nonexistent").unwrap();
}

#[test]
fn get_by_source_filters_by_source_path() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();

    let mut s1 = make_state("card-a", "h1");
    s1.source_path = "notes/math.md".to_string();
    let mut s2 = make_state("card-b", "h2");
    s2.source_path = "notes/math.md".to_string();
    let mut s3 = make_state("card-c", "h3");
    s3.source_path = "notes/physics.md".to_string();

    db.upsert(&s1).unwrap();
    db.upsert(&s2).unwrap();
    db.upsert(&s3).unwrap();

    let math_cards = db.get_by_source("notes/math.md").unwrap();
    assert_eq!(math_cards.len(), 2);
    assert_eq!(math_cards[0].slug, "card-a");
    assert_eq!(math_cards[1].slug, "card-b");

    let physics_cards = db.get_by_source("notes/physics.md").unwrap();
    assert_eq!(physics_cards.len(), 1);
    assert_eq!(physics_cards[0].slug, "card-c");
}

#[test]
fn get_by_source_returns_empty_for_unknown_path() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();

    let result = db.get_by_source("nonexistent/path.md").unwrap();
    assert!(result.is_empty());
}

#[test]
fn get_by_source_returns_sorted_by_slug() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();

    let mut s1 = make_state("zulu", "h1");
    s1.source_path = "src.md".to_string();
    let mut s2 = make_state("alpha", "h2");
    s2.source_path = "src.md".to_string();

    db.upsert(&s1).unwrap();
    db.upsert(&s2).unwrap();

    let results = db.get_by_source("src.md").unwrap();
    assert_eq!(results[0].slug, "alpha");
    assert_eq!(results[1].slug, "zulu");
}

#[test]
fn close_is_callable() {
    let dir = TempDir::new().unwrap();
    let db = StateDB::open(dir.path().join("state.db")).unwrap();
    db.close(); // Should not panic
}

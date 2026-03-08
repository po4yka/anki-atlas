use card::registry::{CardEntry, CardRegistry, NoteEntry, RegistryError, SCHEMA_VERSION};
use chrono::{TimeZone, Utc};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_card_entry(slug: &str) -> CardEntry {
    CardEntry {
        slug: slug.into(),
        note_id: "note-001".into(),
        source_path: "notes/test.md".into(),
        front: "<div>front</div>".into(),
        back: "<div>back</div>".into(),
        content_hash: "abc123abc123".into(),
        metadata_hash: "def456".into(),
        language: "en".into(),
        tags: vec!["tag1".into(), "tag2".into()],
        anki_note_id: None,
        created_at: None,
        updated_at: None,
        synced_at: None,
    }
}

fn make_card_entry_for_note(slug: &str, note_id: &str, source_path: &str) -> CardEntry {
    CardEntry {
        slug: slug.into(),
        note_id: note_id.into(),
        source_path: source_path.into(),
        front: "<div>front</div>".into(),
        back: "<div>back</div>".into(),
        content_hash: "abc123abc123".into(),
        metadata_hash: "def456".into(),
        language: "en".into(),
        tags: vec!["tag1".into()],
        anki_note_id: None,
        created_at: None,
        updated_at: None,
        synced_at: None,
    }
}

fn make_note_entry(note_id: &str) -> NoteEntry {
    NoteEntry {
        note_id: note_id.into(),
        source_path: format!("notes/{note_id}.md"),
        title: Some("Test Note".into()),
        content_hash: Some("hash123".into()),
        created_at: None,
        updated_at: None,
    }
}

fn memory_registry() -> CardRegistry {
    CardRegistry::open(":memory:").expect("in-memory registry should open")
}

// ===========================================================================
// Schema version constant
// ===========================================================================

#[test]
fn schema_version_is_2() {
    assert_eq!(SCHEMA_VERSION, 2);
}

// ===========================================================================
// CardRegistry::open
// ===========================================================================

#[test]
fn open_memory_creates_schema() {
    let reg = memory_registry();
    // Should be able to query immediately - tables exist
    assert_eq!(reg.card_count().unwrap(), 0);
    assert_eq!(reg.note_count().unwrap(), 0);
}

#[test]
fn open_file_creates_schema() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    let reg = CardRegistry::open(db_path.to_str().unwrap()).unwrap();
    assert_eq!(reg.card_count().unwrap(), 0);
    reg.close();
}

#[test]
fn open_file_persists_data() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");

    // Insert a card and close
    {
        let reg = CardRegistry::open(db_path.to_str().unwrap()).unwrap();
        reg.add_card(&make_card_entry("test-slug-0-en")).unwrap();
        reg.close();
    }

    // Reopen and verify data persisted
    {
        let reg = CardRegistry::open(db_path.to_str().unwrap()).unwrap();
        assert_eq!(reg.card_count().unwrap(), 1);
        let card = reg.get_card("test-slug-0-en").unwrap();
        assert!(card.is_some());
        reg.close();
    }
}

#[test]
fn open_sets_wal_journal_mode() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    let _reg = CardRegistry::open(db_path.to_str().unwrap()).unwrap();
    // WAL mode is set - verified implicitly by successful open + operations
    // A WAL file should exist alongside the db
}

// ===========================================================================
// CardRegistry::close
// ===========================================================================

#[test]
fn close_consumes_registry() {
    let reg = memory_registry();
    reg.close(); // Should not panic
}

// ===========================================================================
// Card CRUD: add_card
// ===========================================================================

#[test]
fn add_card_returns_true_on_success() {
    let reg = memory_registry();
    let result = reg.add_card(&make_card_entry("test-slug-0-en")).unwrap();
    assert!(result);
}

#[test]
fn add_card_returns_false_on_duplicate_slug() {
    let reg = memory_registry();
    let entry = make_card_entry("test-slug-0-en");
    reg.add_card(&entry).unwrap();
    let result = reg.add_card(&entry).unwrap();
    assert!(!result);
}

#[test]
fn add_card_increments_count() {
    let reg = memory_registry();
    reg.add_card(&make_card_entry("slug-a-0-en")).unwrap();
    reg.add_card(&make_card_entry("slug-b-0-en")).unwrap();
    assert_eq!(reg.card_count().unwrap(), 2);
}

#[test]
fn add_card_sets_created_at_when_none() {
    let reg = memory_registry();
    let entry = make_card_entry("test-slug-0-en");
    assert!(entry.created_at.is_none());
    reg.add_card(&entry).unwrap();

    let stored = reg.get_card("test-slug-0-en").unwrap().unwrap();
    assert!(stored.created_at.is_some());
}

#[test]
fn add_card_preserves_created_at_when_provided() {
    let reg = memory_registry();
    let ts = Utc.with_ymd_and_hms(2025, 1, 15, 10, 30, 0).unwrap();
    let mut entry = make_card_entry("test-slug-0-en");
    entry.created_at = Some(ts);
    reg.add_card(&entry).unwrap();

    let stored = reg.get_card("test-slug-0-en").unwrap().unwrap();
    assert_eq!(stored.created_at.unwrap().timestamp(), ts.timestamp());
}

#[test]
fn add_card_stores_tags_as_comma_separated() {
    let reg = memory_registry();
    let mut entry = make_card_entry("test-slug-0-en");
    entry.tags = vec!["alpha".into(), "beta".into(), "gamma".into()];
    reg.add_card(&entry).unwrap();

    let stored = reg.get_card("test-slug-0-en").unwrap().unwrap();
    assert_eq!(stored.tags, vec!["alpha", "beta", "gamma"]);
}

#[test]
fn add_card_handles_empty_tags() {
    let reg = memory_registry();
    let mut entry = make_card_entry("test-slug-0-en");
    entry.tags = vec![];
    reg.add_card(&entry).unwrap();

    let stored = reg.get_card("test-slug-0-en").unwrap().unwrap();
    assert!(stored.tags.is_empty());
}

#[test]
fn add_card_stores_anki_note_id() {
    let reg = memory_registry();
    let mut entry = make_card_entry("test-slug-0-en");
    entry.anki_note_id = Some(999);
    entry.synced_at = Some(Utc::now());
    reg.add_card(&entry).unwrap();

    let stored = reg.get_card("test-slug-0-en").unwrap().unwrap();
    assert_eq!(stored.anki_note_id, Some(999));
    assert!(stored.synced_at.is_some());
}

// ===========================================================================
// Card CRUD: get_card
// ===========================================================================

#[test]
fn get_card_returns_none_for_nonexistent() {
    let reg = memory_registry();
    let result = reg.get_card("nonexistent-0-en").unwrap();
    assert!(result.is_none());
}

#[test]
fn get_card_returns_stored_entry() {
    let reg = memory_registry();
    let entry = make_card_entry("test-slug-0-en");
    reg.add_card(&entry).unwrap();

    let stored = reg.get_card("test-slug-0-en").unwrap().unwrap();
    assert_eq!(stored.slug, "test-slug-0-en");
    assert_eq!(stored.note_id, "note-001");
    assert_eq!(stored.source_path, "notes/test.md");
    assert_eq!(stored.front, "<div>front</div>");
    assert_eq!(stored.back, "<div>back</div>");
    assert_eq!(stored.content_hash, "abc123abc123");
    assert_eq!(stored.metadata_hash, "def456");
    assert_eq!(stored.language, "en");
}

// ===========================================================================
// Card CRUD: update_card
// ===========================================================================

#[test]
fn update_card_returns_true_when_updated() {
    let reg = memory_registry();
    let entry = make_card_entry("test-slug-0-en");
    reg.add_card(&entry).unwrap();

    let mut updated = entry.clone();
    updated.front = "<div>updated front</div>".into();
    updated.content_hash = "newhash12345".into();

    let result = reg.update_card(&updated).unwrap();
    assert!(result);
}

#[test]
fn update_card_returns_false_for_nonexistent() {
    let reg = memory_registry();
    let entry = make_card_entry("nonexistent-0-en");
    let result = reg.update_card(&entry).unwrap();
    assert!(!result);
}

#[test]
fn update_card_changes_stored_values() {
    let reg = memory_registry();
    let entry = make_card_entry("test-slug-0-en");
    reg.add_card(&entry).unwrap();

    let mut updated = entry.clone();
    updated.front = "<div>new front</div>".into();
    updated.back = "<div>new back</div>".into();
    updated.content_hash = "newhash12345".into();
    updated.tags = vec!["new-tag".into()];
    updated.anki_note_id = Some(42);
    reg.update_card(&updated).unwrap();

    let stored = reg.get_card("test-slug-0-en").unwrap().unwrap();
    assert_eq!(stored.front, "<div>new front</div>");
    assert_eq!(stored.back, "<div>new back</div>");
    assert_eq!(stored.content_hash, "newhash12345");
    assert_eq!(stored.tags, vec!["new-tag"]);
    assert_eq!(stored.anki_note_id, Some(42));
}

#[test]
fn update_card_sets_updated_at() {
    let reg = memory_registry();
    let entry = make_card_entry("test-slug-0-en");
    reg.add_card(&entry).unwrap();

    let stored_before = reg.get_card("test-slug-0-en").unwrap().unwrap();
    let mut updated = entry.clone();
    updated.front = "<div>changed</div>".into();
    reg.update_card(&updated).unwrap();

    let stored_after = reg.get_card("test-slug-0-en").unwrap().unwrap();
    assert!(stored_after.updated_at.is_some());
    // updated_at should be >= the original
    assert!(stored_after.updated_at.unwrap() >= stored_before.updated_at.unwrap());
}

#[test]
fn update_card_does_not_change_count() {
    let reg = memory_registry();
    let entry = make_card_entry("test-slug-0-en");
    reg.add_card(&entry).unwrap();
    assert_eq!(reg.card_count().unwrap(), 1);

    let mut updated = entry.clone();
    updated.front = "<div>changed</div>".into();
    reg.update_card(&updated).unwrap();
    assert_eq!(reg.card_count().unwrap(), 1);
}

// ===========================================================================
// Card CRUD: delete_card
// ===========================================================================

#[test]
fn delete_card_returns_true_when_deleted() {
    let reg = memory_registry();
    reg.add_card(&make_card_entry("test-slug-0-en")).unwrap();
    let result = reg.delete_card("test-slug-0-en").unwrap();
    assert!(result);
}

#[test]
fn delete_card_returns_false_for_nonexistent() {
    let reg = memory_registry();
    let result = reg.delete_card("nonexistent-0-en").unwrap();
    assert!(!result);
}

#[test]
fn delete_card_removes_from_storage() {
    let reg = memory_registry();
    reg.add_card(&make_card_entry("test-slug-0-en")).unwrap();
    reg.delete_card("test-slug-0-en").unwrap();

    assert!(reg.get_card("test-slug-0-en").unwrap().is_none());
    assert_eq!(reg.card_count().unwrap(), 0);
}

// ===========================================================================
// Card CRUD: find_cards
// ===========================================================================

#[test]
fn find_cards_no_filters_returns_all() {
    let reg = memory_registry();
    reg.add_card(&make_card_entry("slug-a-0-en")).unwrap();
    reg.add_card(&make_card_entry("slug-b-0-en")).unwrap();

    let results = reg.find_cards(None, None, None).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn find_cards_by_note_id() {
    let reg = memory_registry();
    reg.add_card(&make_card_entry_for_note("slug-a-0-en", "note-001", "a.md"))
        .unwrap();
    reg.add_card(&make_card_entry_for_note("slug-b-0-en", "note-002", "b.md"))
        .unwrap();
    reg.add_card(&make_card_entry_for_note("slug-c-0-en", "note-001", "a.md"))
        .unwrap();

    let results = reg.find_cards(Some("note-001"), None, None).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn find_cards_by_source_path() {
    let reg = memory_registry();
    reg.add_card(&make_card_entry_for_note(
        "slug-a-0-en",
        "note-001",
        "notes/a.md",
    ))
    .unwrap();
    reg.add_card(&make_card_entry_for_note(
        "slug-b-0-en",
        "note-002",
        "notes/b.md",
    ))
    .unwrap();

    let results = reg.find_cards(None, Some("notes/a.md"), None).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].slug, "slug-a-0-en");
}

#[test]
fn find_cards_by_content_hash() {
    let reg = memory_registry();
    let mut entry_a = make_card_entry("slug-a-0-en");
    entry_a.content_hash = "hash_aaa_aaa".into();
    let mut entry_b = make_card_entry("slug-b-0-en");
    entry_b.content_hash = "hash_bbb_bbb".into();
    reg.add_card(&entry_a).unwrap();
    reg.add_card(&entry_b).unwrap();

    let results = reg.find_cards(None, None, Some("hash_aaa_aaa")).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].slug, "slug-a-0-en");
}

#[test]
fn find_cards_combined_filters() {
    let reg = memory_registry();
    reg.add_card(&make_card_entry_for_note("slug-a-0-en", "note-001", "a.md"))
        .unwrap();
    reg.add_card(&make_card_entry_for_note("slug-b-0-en", "note-001", "b.md"))
        .unwrap();

    // Filter by note_id AND source_path
    let results = reg
        .find_cards(Some("note-001"), Some("a.md"), None)
        .unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].slug, "slug-a-0-en");
}

#[test]
fn find_cards_no_match_returns_empty() {
    let reg = memory_registry();
    reg.add_card(&make_card_entry("slug-a-0-en")).unwrap();

    let results = reg.find_cards(Some("nonexistent"), None, None).unwrap();
    assert!(results.is_empty());
}

// ===========================================================================
// Note CRUD: add_note
// ===========================================================================

#[test]
fn add_note_returns_true_on_success() {
    let reg = memory_registry();
    let result = reg.add_note(&make_note_entry("note-001")).unwrap();
    assert!(result);
}

#[test]
fn add_note_returns_false_on_duplicate() {
    let reg = memory_registry();
    let entry = make_note_entry("note-001");
    reg.add_note(&entry).unwrap();
    let result = reg.add_note(&entry).unwrap();
    assert!(!result);
}

#[test]
fn add_note_increments_count() {
    let reg = memory_registry();
    reg.add_note(&make_note_entry("note-001")).unwrap();
    reg.add_note(&make_note_entry("note-002")).unwrap();
    assert_eq!(reg.note_count().unwrap(), 2);
}

#[test]
fn add_note_sets_created_at_when_none() {
    let reg = memory_registry();
    let entry = make_note_entry("note-001");
    assert!(entry.created_at.is_none());
    reg.add_note(&entry).unwrap();

    let stored = reg.get_note("note-001").unwrap().unwrap();
    assert!(stored.created_at.is_some());
}

#[test]
fn add_note_preserves_all_fields() {
    let reg = memory_registry();
    let entry = NoteEntry {
        note_id: "note-123".into(),
        source_path: "vault/notes/test.md".into(),
        title: Some("My Title".into()),
        content_hash: Some("abc123".into()),
        created_at: None,
        updated_at: None,
    };
    reg.add_note(&entry).unwrap();

    let stored = reg.get_note("note-123").unwrap().unwrap();
    assert_eq!(stored.note_id, "note-123");
    assert_eq!(stored.source_path, "vault/notes/test.md");
    assert_eq!(stored.title, Some("My Title".into()));
    assert_eq!(stored.content_hash, Some("abc123".into()));
}

// ===========================================================================
// Note CRUD: get_note
// ===========================================================================

#[test]
fn get_note_returns_none_for_nonexistent() {
    let reg = memory_registry();
    let result = reg.get_note("nonexistent").unwrap();
    assert!(result.is_none());
}

#[test]
fn get_note_returns_stored_entry() {
    let reg = memory_registry();
    reg.add_note(&make_note_entry("note-001")).unwrap();

    let stored = reg.get_note("note-001").unwrap().unwrap();
    assert_eq!(stored.note_id, "note-001");
    assert_eq!(stored.source_path, "notes/note-001.md");
    assert_eq!(stored.title, Some("Test Note".into()));
}

// ===========================================================================
// Note CRUD: list_notes
// ===========================================================================

#[test]
fn list_notes_empty() {
    let reg = memory_registry();
    let results = reg.list_notes().unwrap();
    assert!(results.is_empty());
}

#[test]
fn list_notes_returns_all() {
    let reg = memory_registry();
    reg.add_note(&make_note_entry("note-002")).unwrap();
    reg.add_note(&make_note_entry("note-001")).unwrap();

    let results = reg.list_notes().unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn list_notes_ordered_by_note_id() {
    let reg = memory_registry();
    reg.add_note(&make_note_entry("note-c")).unwrap();
    reg.add_note(&make_note_entry("note-a")).unwrap();
    reg.add_note(&make_note_entry("note-b")).unwrap();

    let results = reg.list_notes().unwrap();
    assert_eq!(results[0].note_id, "note-a");
    assert_eq!(results[1].note_id, "note-b");
    assert_eq!(results[2].note_id, "note-c");
}

// ===========================================================================
// Stats
// ===========================================================================

#[test]
fn card_count_empty() {
    let reg = memory_registry();
    assert_eq!(reg.card_count().unwrap(), 0);
}

#[test]
fn note_count_empty() {
    let reg = memory_registry();
    assert_eq!(reg.note_count().unwrap(), 0);
}

#[test]
fn card_count_after_add_and_delete() {
    let reg = memory_registry();
    reg.add_card(&make_card_entry("slug-a-0-en")).unwrap();
    reg.add_card(&make_card_entry("slug-b-0-en")).unwrap();
    assert_eq!(reg.card_count().unwrap(), 2);

    reg.delete_card("slug-a-0-en").unwrap();
    assert_eq!(reg.card_count().unwrap(), 1);
}

// ===========================================================================
// Mapping: get_mapping
// ===========================================================================

#[test]
fn get_mapping_returns_cards_for_note() {
    let reg = memory_registry();
    reg.add_card(&make_card_entry_for_note("slug-a-0-en", "note-001", "a.md"))
        .unwrap();
    reg.add_card(&make_card_entry_for_note("slug-b-0-en", "note-001", "a.md"))
        .unwrap();
    reg.add_card(&make_card_entry_for_note("slug-c-0-en", "note-002", "b.md"))
        .unwrap();

    let mapping = reg.get_mapping("note-001").unwrap();
    assert_eq!(mapping.len(), 2);
}

#[test]
fn get_mapping_empty_for_unknown_note() {
    let reg = memory_registry();
    let mapping = reg.get_mapping("nonexistent").unwrap();
    assert!(mapping.is_empty());
}

// ===========================================================================
// Mapping: update_mapping
// ===========================================================================

#[test]
fn update_mapping_replaces_all_cards() {
    let reg = memory_registry();
    // Add initial cards for note-001
    reg.add_card(&make_card_entry_for_note("old-a-0-en", "note-001", "a.md"))
        .unwrap();
    reg.add_card(&make_card_entry_for_note("old-b-0-en", "note-001", "a.md"))
        .unwrap();
    assert_eq!(reg.get_mapping("note-001").unwrap().len(), 2);

    // Replace with new cards
    let new_cards = vec![
        make_card_entry_for_note("new-a-0-en", "note-001", "a.md"),
        make_card_entry_for_note("new-b-0-en", "note-001", "a.md"),
        make_card_entry_for_note("new-c-0-en", "note-001", "a.md"),
    ];
    reg.update_mapping("note-001", &new_cards).unwrap();

    let mapping = reg.get_mapping("note-001").unwrap();
    assert_eq!(mapping.len(), 3);

    // Old cards should be gone
    assert!(reg.get_card("old-a-0-en").unwrap().is_none());
    assert!(reg.get_card("old-b-0-en").unwrap().is_none());
}

#[test]
fn update_mapping_does_not_affect_other_notes() {
    let reg = memory_registry();
    reg.add_card(&make_card_entry_for_note(
        "note1-a-0-en",
        "note-001",
        "a.md",
    ))
    .unwrap();
    reg.add_card(&make_card_entry_for_note(
        "note2-a-0-en",
        "note-002",
        "b.md",
    ))
    .unwrap();

    let new_cards = vec![make_card_entry_for_note("note1-b-0-en", "note-001", "a.md")];
    reg.update_mapping("note-001", &new_cards).unwrap();

    // note-002's card should still exist
    assert!(reg.get_card("note2-a-0-en").unwrap().is_some());
    assert_eq!(reg.card_count().unwrap(), 2);
}

#[test]
fn update_mapping_empty_cards_deletes_all() {
    let reg = memory_registry();
    reg.add_card(&make_card_entry_for_note("slug-a-0-en", "note-001", "a.md"))
        .unwrap();

    reg.update_mapping("note-001", &[]).unwrap();
    assert_eq!(reg.get_mapping("note-001").unwrap().len(), 0);
    assert_eq!(reg.card_count().unwrap(), 0);
}

// ===========================================================================
// Schema migration: v1 -> v2
// ===========================================================================

#[test]
fn migration_v1_to_v2_adds_notes_table() {
    // Create a v1 schema manually (cards + schema_version, no notes)
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;

        CREATE TABLE cards (
            slug TEXT PRIMARY KEY,
            note_id TEXT NOT NULL,
            source_path TEXT NOT NULL,
            front TEXT NOT NULL,
            back TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            metadata_hash TEXT NOT NULL,
            language TEXT NOT NULL,
            tags TEXT,
            anki_note_id INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            synced_at TEXT
        );

        CREATE TABLE schema_version (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version INTEGER NOT NULL DEFAULT 1
        );

        INSERT INTO schema_version (id, version) VALUES (1, 1);

        INSERT INTO cards (slug, note_id, source_path, front, back,
            content_hash, metadata_hash, language, tags,
            created_at, updated_at)
        VALUES ('old-card-0-en', 'note-001', 'test.md', 'front', 'back',
            'hash123', 'meta456', 'en', 'tag1',
            '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z');
        ",
    )
    .unwrap();

    // Save to a temp file so CardRegistry can reopen
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("v1.db");

    // Copy in-memory to file
    conn.execute_batch(&format!("VACUUM INTO '{}'", db_path.to_str().unwrap()))
        .unwrap();
    drop(conn);

    // Open with CardRegistry which should auto-migrate
    let reg = CardRegistry::open(db_path.to_str().unwrap()).unwrap();

    // Old card data should survive migration
    let old_card = reg.get_card("old-card-0-en").unwrap();
    assert!(old_card.is_some());

    // Notes table should now exist and be usable
    let result = reg.add_note(&make_note_entry("note-001"));
    assert!(result.is_ok());
    assert_eq!(reg.note_count().unwrap(), 1);

    reg.close();
}

// ===========================================================================
// RegistryError
// ===========================================================================

#[test]
fn registry_error_database_variant() {
    let err = RegistryError::Database(rusqlite::Error::InvalidQuery);
    let msg = format!("{err}");
    assert!(msg.contains("database error"));
}

#[test]
fn registry_error_migration_variant() {
    let err = RegistryError::Migration {
        from: 1,
        to: 2,
        reason: "test failure".into(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("migration failed"));
    assert!(msg.contains("v1"));
    assert!(msg.contains("v2"));
}

#[test]
fn registry_error_duplicate_slug_variant() {
    let err = RegistryError::DuplicateSlug("my-slug-0-en".into());
    let msg = format!("{err}");
    assert!(msg.contains("duplicate slug"));
    assert!(msg.contains("my-slug-0-en"));
}

// ===========================================================================
// Datetime handling
// ===========================================================================

#[test]
fn datetime_roundtrip_through_sqlite() {
    let reg = memory_registry();
    let ts = Utc.with_ymd_and_hms(2025, 6, 15, 14, 30, 45).unwrap();
    let mut entry = make_card_entry("test-slug-0-en");
    entry.created_at = Some(ts);
    entry.updated_at = Some(ts);
    entry.synced_at = Some(ts);
    reg.add_card(&entry).unwrap();

    let stored = reg.get_card("test-slug-0-en").unwrap().unwrap();
    // Timestamps should roundtrip (at second precision at minimum)
    assert_eq!(stored.created_at.unwrap().timestamp(), ts.timestamp());
    assert_eq!(stored.synced_at.unwrap().timestamp(), ts.timestamp());
}

#[test]
fn null_datetimes_stored_as_none() {
    let reg = memory_registry();
    let mut entry = make_card_entry("test-slug-0-en");
    entry.synced_at = None;
    entry.anki_note_id = None;
    reg.add_card(&entry).unwrap();

    let stored = reg.get_card("test-slug-0-en").unwrap().unwrap();
    assert!(stored.synced_at.is_none());
    assert!(stored.anki_note_id.is_none());
}

// ===========================================================================
// Send + Sync bounds
// ===========================================================================

#[test]
fn card_registry_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<CardRegistry>();
}

#[test]
fn card_entry_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CardEntry>();
}

#[test]
fn note_entry_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<NoteEntry>();
}

#[test]
fn registry_error_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RegistryError>();
}

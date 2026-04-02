use card::mapping::{CardMappingEntry, NoteMapping};
use card::registry::CardEntry;
use chrono::{TimeZone, Utc};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_card_entry(slug: &str, language: &str, anki_note_id: Option<i64>) -> CardEntry {
    let synced_at = anki_note_id.map(|_| Utc::now());
    CardEntry {
        slug: slug.into(),
        note_id: "note-001".into(),
        source_path: "notes/test.md".into(),
        front: "<div>front</div>".into(),
        back: "<div>back</div>".into(),
        content_hash: "abc123abc123".into(),
        metadata_hash: "def456".into(),
        language: language.into(),
        tags: vec!["tag1".into()],
        anki_note_id,
        created_at: Some(Utc::now()),
        updated_at: Some(Utc::now()),
        synced_at,
    }
}

fn synced_mapping_entry(slug: &str) -> CardMappingEntry {
    CardMappingEntry {
        slug: slug.into(),
        language: "en".into(),
        anki_note_id: Some(12345),
        synced_at: Some(Utc::now()),
        content_hash: "abc123abc123".into(),
    }
}

fn unsynced_mapping_entry(slug: &str) -> CardMappingEntry {
    CardMappingEntry {
        slug: slug.into(),
        language: "en".into(),
        anki_note_id: None,
        synced_at: None,
        content_hash: "abc123abc123".into(),
    }
}

// ===========================================================================
// CardMappingEntry::is_synced
// ===========================================================================

#[test]
fn is_synced_returns_true_when_anki_note_id_present() {
    let entry = synced_mapping_entry("test-card-0-en");
    assert!(entry.is_synced());
}

#[test]
fn is_synced_returns_false_when_anki_note_id_none() {
    let entry = unsynced_mapping_entry("test-card-0-en");
    assert!(!entry.is_synced());
}

#[test]
fn is_synced_with_zero_note_id_still_true() {
    let entry = CardMappingEntry {
        slug: "test-0-en".into(),
        language: "en".into(),
        anki_note_id: Some(0),
        synced_at: None,
        content_hash: "hash12".into(),
    };
    assert!(entry.is_synced());
}

// ===========================================================================
// CardMappingEntry::from_card_entry
// ===========================================================================

#[test]
fn from_card_entry_copies_slug() {
    let ce = make_card_entry("my-slug-0-en", "en", None);
    let me = CardMappingEntry::from_card_entry(&ce);
    assert_eq!(me.slug, "my-slug-0-en");
}

#[test]
fn from_card_entry_copies_language() {
    let ce = make_card_entry("test-0-ru", "ru", None);
    let me = CardMappingEntry::from_card_entry(&ce);
    assert_eq!(me.language, "ru");
}

#[test]
fn from_card_entry_copies_anki_note_id_some() {
    let ce = make_card_entry("test-0-en", "en", Some(999));
    let me = CardMappingEntry::from_card_entry(&ce);
    assert_eq!(me.anki_note_id, Some(999));
}

#[test]
fn from_card_entry_copies_anki_note_id_none() {
    let ce = make_card_entry("test-0-en", "en", None);
    let me = CardMappingEntry::from_card_entry(&ce);
    assert_eq!(me.anki_note_id, None);
}

#[test]
fn from_card_entry_copies_synced_at() {
    let ts = Utc.with_ymd_and_hms(2026, 1, 15, 10, 30, 0).unwrap();
    let mut ce = make_card_entry("test-0-en", "en", Some(1));
    ce.synced_at = Some(ts);
    let me = CardMappingEntry::from_card_entry(&ce);
    assert_eq!(me.synced_at, Some(ts));
}

#[test]
fn from_card_entry_copies_content_hash() {
    let ce = make_card_entry("test-0-en", "en", None);
    let me = CardMappingEntry::from_card_entry(&ce);
    assert_eq!(me.content_hash, "abc123abc123");
}

#[test]
fn from_card_entry_synced_entry_is_synced() {
    let ce = make_card_entry("test-0-en", "en", Some(42));
    let me = CardMappingEntry::from_card_entry(&ce);
    assert!(me.is_synced());
}

#[test]
fn from_card_entry_unsynced_entry_not_synced() {
    let ce = make_card_entry("test-0-en", "en", None);
    let me = CardMappingEntry::from_card_entry(&ce);
    assert!(!me.is_synced());
}

// ===========================================================================
// NoteMapping::card_count
// ===========================================================================

#[test]
fn card_count_empty() {
    let nm = NoteMapping {
        note_path: "notes/test.md".into(),
        note_id: "note-001".into(),
        note_title: "Test Note".into(),
        cards: vec![],
        last_sync: None,
        is_orphan: false,
    };
    assert_eq!(nm.card_count(), 0);
}

#[test]
fn card_count_single() {
    let nm = NoteMapping {
        note_path: "notes/test.md".into(),
        note_id: "note-001".into(),
        note_title: "Test Note".into(),
        cards: vec![synced_mapping_entry("card-0-en")],
        last_sync: None,
        is_orphan: false,
    };
    assert_eq!(nm.card_count(), 1);
}

#[test]
fn card_count_multiple() {
    let nm = NoteMapping {
        note_path: "notes/test.md".into(),
        note_id: "note-001".into(),
        note_title: "Test Note".into(),
        cards: vec![
            synced_mapping_entry("card-0-en"),
            unsynced_mapping_entry("card-1-en"),
            synced_mapping_entry("card-2-en"),
        ],
        last_sync: None,
        is_orphan: false,
    };
    assert_eq!(nm.card_count(), 3);
}

// ===========================================================================
// NoteMapping::synced_count
// ===========================================================================

#[test]
fn synced_count_none_synced() {
    let nm = NoteMapping {
        note_path: "notes/test.md".into(),
        note_id: "note-001".into(),
        note_title: "Test Note".into(),
        cards: vec![
            unsynced_mapping_entry("card-0-en"),
            unsynced_mapping_entry("card-1-en"),
        ],
        last_sync: None,
        is_orphan: false,
    };
    assert_eq!(nm.synced_count(), 0);
}

#[test]
fn synced_count_all_synced() {
    let nm = NoteMapping {
        note_path: "notes/test.md".into(),
        note_id: "note-001".into(),
        note_title: "Test Note".into(),
        cards: vec![
            synced_mapping_entry("card-0-en"),
            synced_mapping_entry("card-1-en"),
        ],
        last_sync: None,
        is_orphan: false,
    };
    assert_eq!(nm.synced_count(), 2);
}

#[test]
fn synced_count_mixed() {
    let nm = NoteMapping {
        note_path: "notes/test.md".into(),
        note_id: "note-001".into(),
        note_title: "Test Note".into(),
        cards: vec![
            synced_mapping_entry("card-0-en"),
            unsynced_mapping_entry("card-1-en"),
            synced_mapping_entry("card-2-en"),
        ],
        last_sync: None,
        is_orphan: false,
    };
    assert_eq!(nm.synced_count(), 2);
}

#[test]
fn synced_count_empty_cards() {
    let nm = NoteMapping {
        note_path: "notes/test.md".into(),
        note_id: "note-001".into(),
        note_title: "Test Note".into(),
        cards: vec![],
        last_sync: None,
        is_orphan: false,
    };
    assert_eq!(nm.synced_count(), 0);
}

// ===========================================================================
// NoteMapping::unsynced_count
// ===========================================================================

#[test]
fn unsynced_count_none_synced() {
    let nm = NoteMapping {
        note_path: "notes/test.md".into(),
        note_id: "note-001".into(),
        note_title: "Test Note".into(),
        cards: vec![
            unsynced_mapping_entry("card-0-en"),
            unsynced_mapping_entry("card-1-en"),
            unsynced_mapping_entry("card-2-en"),
        ],
        last_sync: None,
        is_orphan: false,
    };
    assert_eq!(nm.unsynced_count(), 3);
}

#[test]
fn unsynced_count_all_synced() {
    let nm = NoteMapping {
        note_path: "notes/test.md".into(),
        note_id: "note-001".into(),
        note_title: "Test Note".into(),
        cards: vec![
            synced_mapping_entry("card-0-en"),
            synced_mapping_entry("card-1-en"),
        ],
        last_sync: None,
        is_orphan: false,
    };
    assert_eq!(nm.unsynced_count(), 0);
}

#[test]
fn unsynced_count_mixed() {
    let nm = NoteMapping {
        note_path: "notes/test.md".into(),
        note_id: "note-001".into(),
        note_title: "Test Note".into(),
        cards: vec![
            synced_mapping_entry("card-0-en"),
            unsynced_mapping_entry("card-1-en"),
            synced_mapping_entry("card-2-en"),
            unsynced_mapping_entry("card-3-en"),
        ],
        last_sync: None,
        is_orphan: false,
    };
    assert_eq!(nm.unsynced_count(), 2);
}

#[test]
fn unsynced_equals_card_count_minus_synced() {
    let nm = NoteMapping {
        note_path: "notes/test.md".into(),
        note_id: "note-001".into(),
        note_title: "Test Note".into(),
        cards: vec![
            synced_mapping_entry("card-0-en"),
            unsynced_mapping_entry("card-1-en"),
            synced_mapping_entry("card-2-en"),
        ],
        last_sync: None,
        is_orphan: false,
    };
    assert_eq!(nm.unsynced_count(), nm.card_count() - nm.synced_count());
}

// ===========================================================================
// NoteMapping fields
// ===========================================================================

#[test]
fn note_mapping_stores_note_path() {
    let nm = NoteMapping {
        note_path: "vault/notes/rust.md".into(),
        note_id: "note-42".into(),
        note_title: "Rust Basics".into(),
        cards: vec![],
        last_sync: None,
        is_orphan: false,
    };
    assert_eq!(nm.note_path, "vault/notes/rust.md");
    assert_eq!(nm.note_id, "note-42");
    assert_eq!(nm.note_title, "Rust Basics");
}

#[test]
fn note_mapping_orphan_flag() {
    let nm = NoteMapping {
        note_path: "deleted/note.md".into(),
        note_id: "note-99".into(),
        note_title: "Old Note".into(),
        cards: vec![synced_mapping_entry("old-card-0-en")],
        last_sync: None,
        is_orphan: true,
    };
    assert!(nm.is_orphan);
    assert_eq!(nm.card_count(), 1);
}

#[test]
fn note_mapping_last_sync() {
    let ts = Utc.with_ymd_and_hms(2026, 3, 8, 12, 0, 0).unwrap();
    let nm = NoteMapping {
        note_path: "notes/test.md".into(),
        note_id: "note-001".into(),
        note_title: "Test".into(),
        cards: vec![],
        last_sync: Some(ts),
        is_orphan: false,
    };
    assert_eq!(nm.last_sync, Some(ts));
}

// ===========================================================================
// CardMappingEntry fields and serialization
// ===========================================================================

#[test]
fn card_mapping_entry_fields() {
    let ts = Utc.with_ymd_and_hms(2026, 2, 1, 8, 0, 0).unwrap();
    let entry = CardMappingEntry {
        slug: "rust-ownership-0-en".into(),
        language: "en".into(),
        anki_note_id: Some(54321),
        synced_at: Some(ts),
        content_hash: "fedcba987654".into(),
    };
    assert_eq!(entry.slug, "rust-ownership-0-en");
    assert_eq!(entry.language, "en");
    assert_eq!(entry.anki_note_id, Some(54321));
    assert_eq!(entry.synced_at, Some(ts));
    assert_eq!(entry.content_hash, "fedcba987654");
}

// ===========================================================================
// Send + Sync bounds
// ===========================================================================

common::assert_send_sync!(CardMappingEntry, NoteMapping);

// ===========================================================================
// Clone and PartialEq for CardMappingEntry
// ===========================================================================

#[test]
fn card_mapping_entry_clone_eq() {
    let entry = synced_mapping_entry("test-0-en");
    let cloned = entry.clone();
    assert_eq!(entry, cloned);
}

use anki_reader::reader::{AnkiReader, read_anki_collection};
use common::{CardId, DeckId, NoteId};
use rusqlite::Connection;
use tempfile::NamedTempFile;

/// Helper: create a minimal Anki SQLite database with legacy schema.
fn create_legacy_db() -> NamedTempFile {
    let file = NamedTempFile::new().expect("create temp file");
    let conn = Connection::open(file.path()).expect("open sqlite");

    // Legacy schema: col table with JSON blobs for decks and models
    conn.execute_batch(
        "
        CREATE TABLE col (
            id INTEGER PRIMARY KEY,
            crt INTEGER,
            mod INTEGER,
            scm INTEGER,
            ver INTEGER,
            dty INTEGER,
            usn INTEGER,
            ls INTEGER,
            conf TEXT,
            models TEXT,
            decks TEXT,
            dconf TEXT,
            tags TEXT
        );
        INSERT INTO col VALUES (
            1, 0, 0, 0, 11, 0, 0, 0, '{}',
            '{\"1234567890\": {\"id\": 1234567890, \"name\": \"Basic\", \"flds\": [{\"name\": \"Front\", \"ord\": 0}, {\"name\": \"Back\", \"ord\": 1}], \"tmpls\": [{\"name\": \"Card 1\"}]}}',
            '{\"1\": {\"id\": 1, \"name\": \"Default\"}, \"2\": {\"id\": 2, \"name\": \"Japanese::Vocab\"}}',
            '{}', '{}'
        );

        CREATE TABLE notes (
            id INTEGER PRIMARY KEY,
            guid TEXT,
            mid INTEGER,
            mod INTEGER,
            usn INTEGER,
            tags TEXT,
            flds TEXT,
            sfld TEXT,
            csum INTEGER,
            flags INTEGER,
            data TEXT
        );
        INSERT INTO notes VALUES (100, 'abc', 1234567890, 1700000000, -1, ' vocab japanese ', 'Hello\x1fWorld', 'Hello', 0, 0, '');
        INSERT INTO notes VALUES (101, 'def', 1234567890, 1700000001, -1, '', 'Question\x1fAnswer', 'Question', 0, 0, '');

        CREATE TABLE cards (
            id INTEGER PRIMARY KEY,
            nid INTEGER,
            did INTEGER,
            ord INTEGER,
            mod INTEGER,
            usn INTEGER,
            type INTEGER,
            queue INTEGER,
            due INTEGER,
            ivl INTEGER,
            factor INTEGER,
            reps INTEGER,
            lapses INTEGER,
            left INTEGER,
            odue INTEGER,
            odid INTEGER,
            flags INTEGER,
            data TEXT
        );
        INSERT INTO cards VALUES (500, 100, 1, 0, 1700000000, -1, 2, 2, 1000, 30, 2500, 10, 2, 0, 0, 0, 0, '');
        INSERT INTO cards VALUES (501, 101, 2, 0, 1700000001, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '');

        CREATE TABLE revlog (
            id INTEGER PRIMARY KEY,
            cid INTEGER,
            usn INTEGER,
            ease INTEGER,
            ivl INTEGER,
            lastIvl INTEGER,
            factor INTEGER,
            time INTEGER,
            type INTEGER
        );
        INSERT INTO revlog VALUES (1700000000000, 500, -1, 3, 30, 15, 2500, 8000, 1);
        INSERT INTO revlog VALUES (1700000001000, 500, -1, 1, 0, 30, 2100, 15000, 2);
        INSERT INTO revlog VALUES (1700000002000, 500, -1, 3, 10, 0, 2200, 6000, 1);
        ",
    )
    .expect("create legacy db schema");

    file
}

/// Helper: create a minimal Anki SQLite database with modern schema.
fn create_modern_db() -> NamedTempFile {
    let file = NamedTempFile::new().expect("create temp file");
    let conn = Connection::open(file.path()).expect("open sqlite");

    conn.execute_batch(
        "
        CREATE TABLE notetypes (
            id INTEGER PRIMARY KEY,
            name TEXT,
            mtime_secs INTEGER,
            usn INTEGER,
            config BLOB
        );
        INSERT INTO notetypes VALUES (999, 'Cloze', 0, 0, X'');

        CREATE TABLE fields (
            ntid INTEGER,
            ord INTEGER,
            name TEXT,
            config BLOB
        );
        INSERT INTO fields VALUES (999, 0, 'Text', X'');
        INSERT INTO fields VALUES (999, 1, 'Extra', X'');

        CREATE TABLE templates (
            ntid INTEGER,
            ord INTEGER,
            name TEXT,
            mtime_secs INTEGER,
            usn INTEGER,
            config BLOB
        );
        INSERT INTO templates VALUES (999, 0, 'Cloze', 0, 0, X'');

        CREATE TABLE decks (
            id INTEGER PRIMARY KEY,
            name TEXT,
            mtime_secs INTEGER,
            usn INTEGER,
            common BLOB
        );
        INSERT INTO decks VALUES (1, 'Default', 0, 0, X'');
        INSERT INTO decks VALUES (10, 'Science::Physics', 0, 0, X'');

        CREATE TABLE notes (
            id INTEGER PRIMARY KEY,
            guid TEXT,
            mid INTEGER,
            mod INTEGER,
            usn INTEGER,
            tags TEXT,
            flds TEXT,
            sfld TEXT,
            csum INTEGER,
            flags INTEGER,
            data TEXT
        );
        INSERT INTO notes VALUES (200, 'ghi', 999, 1700000000, -1, ' physics ', '{{c1::gravity}} pulls\x1fNewton', 'gravity pulls', 0, 0, '');

        CREATE TABLE cards (
            id INTEGER PRIMARY KEY,
            nid INTEGER,
            did INTEGER,
            ord INTEGER,
            mod INTEGER,
            usn INTEGER,
            type INTEGER,
            queue INTEGER,
            due INTEGER,
            ivl INTEGER,
            factor INTEGER,
            reps INTEGER,
            lapses INTEGER,
            left INTEGER,
            odue INTEGER,
            odid INTEGER,
            flags INTEGER,
            data TEXT
        );
        INSERT INTO cards VALUES (600, 200, 10, 0, 1700000000, -1, 1, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, '');

        CREATE TABLE revlog (
            id INTEGER PRIMARY KEY,
            cid INTEGER,
            usn INTEGER,
            ease INTEGER,
            ivl INTEGER,
            lastIvl INTEGER,
            factor INTEGER,
            time INTEGER,
            type INTEGER
        );
        INSERT INTO revlog VALUES (1700000000000, 600, -1, 3, 1, 0, 0, 5000, 0);
        ",
    )
    .expect("create modern db schema");

    file
}

// --- AnkiReader::new ---

#[test]
fn reader_new_validates_file_exists() {
    let result = AnkiReader::new("/nonexistent/path/collection.anki2");
    assert!(result.is_err());
}

#[test]
fn reader_new_accepts_valid_path() {
    let db = create_legacy_db();
    let result = AnkiReader::new(db.path());
    assert!(result.is_ok());
}

// --- Legacy schema tests ---

#[test]
fn reader_reads_legacy_decks() {
    let db = create_legacy_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();

    let decks = reader.read_decks().unwrap();
    assert_eq!(decks.len(), 2);

    let names: Vec<&str> = decks.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"Default"));
    assert!(names.contains(&"Japanese::Vocab"));

    // Check parent extraction for nested deck
    let nested = decks.iter().find(|d| d.name == "Japanese::Vocab").unwrap();
    assert_eq!(nested.parent_name.as_deref(), Some("Japanese"));
}

#[test]
fn reader_reads_legacy_models() {
    let db = create_legacy_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();

    let models = reader.read_models().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "Basic");
    assert_eq!(models[0].fields.len(), 2);
}

#[test]
fn reader_reads_notes_with_field_splitting() {
    let db = create_legacy_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();

    let models = reader.read_models().unwrap();
    let notes = reader.read_notes(&models).unwrap();
    assert_eq!(notes.len(), 2);

    let note = notes.iter().find(|n| n.note_id == NoteId(100)).unwrap();
    // Fields split on \x1f
    assert_eq!(note.fields, vec!["Hello", "World"]);
    // Named field mapping
    assert_eq!(note.fields_json["Front"], "Hello");
    assert_eq!(note.fields_json["Back"], "World");
    // Tags parsed from space-separated string
    assert!(note.tags.contains(&"vocab".to_string()));
    assert!(note.tags.contains(&"japanese".to_string()));
}

#[test]
fn reader_reads_notes_empty_tags() {
    let db = create_legacy_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();

    let models = reader.read_models().unwrap();
    let notes = reader.read_notes(&models).unwrap();
    let note = notes.iter().find(|n| n.note_id == NoteId(101)).unwrap();
    assert!(note.tags.is_empty());
}

#[test]
fn reader_reads_cards() {
    let db = create_legacy_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();

    let cards = reader.read_cards().unwrap();
    assert_eq!(cards.len(), 2);

    let card = cards.iter().find(|c| c.card_id == CardId(500)).unwrap();
    assert_eq!(card.note_id, NoteId(100));
    assert_eq!(card.deck_id, DeckId(1));
    assert_eq!(card.ease, 2500);
    assert_eq!(card.ivl, 30);
    assert_eq!(card.queue, 2);
    assert_eq!(card.card_type, 2);
}

#[test]
fn reader_reads_revlog() {
    let db = create_legacy_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();

    let revlog = reader.read_revlog().unwrap();
    assert_eq!(revlog.len(), 3);

    let entry = revlog.iter().find(|r| r.id == 1700000000000).unwrap();
    assert_eq!(entry.card_id, CardId(500));
    assert_eq!(entry.button_chosen, 3);
    assert_eq!(entry.time_ms, 8000);
    assert_eq!(entry.review_type, 1);
}

#[test]
fn reader_computes_card_stats() {
    let db = create_legacy_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();

    let stats = reader.compute_card_stats().unwrap();
    assert_eq!(stats.len(), 1); // Only card 500 has revlog entries

    let stat = &stats[0];
    assert_eq!(stat.card_id, CardId(500));
    assert_eq!(stat.reviews, 3);
    // fail_rate: 1 fail out of 3 reviews
    assert!((stat.fail_rate.unwrap() - 1.0 / 3.0).abs() < 0.01);
    assert!(stat.last_review_at.is_some());
    assert_eq!(stat.total_time_ms, 8000 + 15000 + 6000);
}

// --- Modern schema tests ---

#[test]
fn reader_reads_modern_decks() {
    let db = create_modern_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();

    let decks = reader.read_decks().unwrap();
    assert_eq!(decks.len(), 2);

    let names: Vec<&str> = decks.iter().map(|d| d.name.as_str()).collect();
    assert!(names.contains(&"Default"));
    assert!(names.contains(&"Science::Physics"));
}

#[test]
fn reader_reads_modern_models() {
    let db = create_modern_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();

    let models = reader.read_models().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].name, "Cloze");
    assert_eq!(models[0].fields.len(), 2);
}

#[test]
fn reader_reads_modern_notes() {
    let db = create_modern_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();

    let models = reader.read_models().unwrap();
    let notes = reader.read_notes(&models).unwrap();
    assert_eq!(notes.len(), 1);
    assert_eq!(notes[0].note_id, NoteId(200));
    assert!(notes[0].tags.contains(&"physics".to_string()));
    // Field split on \x1f
    assert_eq!(notes[0].fields.len(), 2);
    assert!(notes[0].fields[0].contains("gravity"));
}

// --- read_collection (full pipeline) ---

#[test]
fn read_collection_legacy() {
    let db = create_legacy_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();

    let collection = reader.read_collection().unwrap();
    assert_eq!(collection.decks.len(), 2);
    assert_eq!(collection.models.len(), 1);
    assert_eq!(collection.notes.len(), 2);
    assert_eq!(collection.cards.len(), 2);
    assert_eq!(collection.card_stats.len(), 1);
    assert_eq!(collection.schema_version, 11);
}

#[test]
fn read_collection_modern() {
    let db = create_modern_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();

    let collection = reader.read_collection().unwrap();
    assert_eq!(collection.decks.len(), 2);
    assert_eq!(collection.models.len(), 1);
    assert_eq!(collection.notes.len(), 1);
    assert_eq!(collection.cards.len(), 1);
}

// --- Convenience function ---

#[test]
fn read_anki_collection_convenience() {
    let db = create_legacy_db();
    let collection = read_anki_collection(db.path()).unwrap();
    assert_eq!(collection.decks.len(), 2);
    assert_eq!(collection.notes.len(), 2);
}

// --- Temp file management ---

#[test]
fn reader_temp_file_created_on_open() {
    let db = create_legacy_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();
    // Reader should be usable after open
    let decks = reader.read_decks().unwrap();
    assert!(!decks.is_empty());
}

#[test]
fn reader_close_cleanup() {
    let db = create_legacy_db();
    let mut reader = AnkiReader::new(db.path()).unwrap();
    reader.open().unwrap();
    reader.close();
    // After close, operations should fail or reader should be unusable
}

#[test]
fn reader_drop_cleanup() {
    let db = create_legacy_db();
    {
        let mut reader = AnkiReader::new(db.path()).unwrap();
        reader.open().unwrap();
        // Drop at end of scope should clean up
    }
    // No panic = success
}

// --- Edge cases ---

#[test]
fn reader_empty_database() {
    let file = NamedTempFile::new().expect("create temp file");
    let conn = Connection::open(file.path()).expect("open sqlite");
    conn.execute_batch(
        "
        CREATE TABLE col (
            id INTEGER PRIMARY KEY,
            crt INTEGER,
            mod INTEGER,
            scm INTEGER,
            ver INTEGER,
            dty INTEGER,
            usn INTEGER,
            ls INTEGER,
            conf TEXT,
            models TEXT,
            decks TEXT,
            dconf TEXT,
            tags TEXT
        );
        INSERT INTO col VALUES (1, 0, 0, 0, 11, 0, 0, 0, '{}', '{}', '{}', '{}', '{}');
        CREATE TABLE notes (id INTEGER PRIMARY KEY, guid TEXT, mid INTEGER, mod INTEGER, usn INTEGER, tags TEXT, flds TEXT, sfld TEXT, csum INTEGER, flags INTEGER, data TEXT);
        CREATE TABLE cards (id INTEGER PRIMARY KEY, nid INTEGER, did INTEGER, ord INTEGER, mod INTEGER, usn INTEGER, type INTEGER, queue INTEGER, due INTEGER, ivl INTEGER, factor INTEGER, reps INTEGER, lapses INTEGER, left INTEGER, odue INTEGER, odid INTEGER, flags INTEGER, data TEXT);
        CREATE TABLE revlog (id INTEGER PRIMARY KEY, cid INTEGER, usn INTEGER, ease INTEGER, ivl INTEGER, lastIvl INTEGER, factor INTEGER, time INTEGER, type INTEGER);
        ",
    )
    .unwrap();

    let mut reader = AnkiReader::new(file.path()).unwrap();
    reader.open().unwrap();

    let collection = reader.read_collection().unwrap();
    assert!(collection.decks.is_empty());
    assert!(collection.models.is_empty());
    assert!(collection.notes.is_empty());
    assert!(collection.cards.is_empty());
    assert!(collection.card_stats.is_empty());
}

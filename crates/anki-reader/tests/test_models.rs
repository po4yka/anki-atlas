use anki_reader::models::{
    AnkiCard, AnkiCollection, AnkiDeck, AnkiModel, AnkiNote, AnkiRevlogEntry, CardStats,
};
use chrono::Utc;
use common::{CardId, DeckId, ModelId, NoteId};
use std::collections::HashMap;

// --- AnkiDeck ---

#[test]
fn deck_create_with_all_fields() {
    let deck = AnkiDeck {
        deck_id: DeckId(1234),
        name: "Japanese::Vocab".into(),
        parent_name: Some("Japanese".into()),
        config: serde_json::json!({"newToday": [0, 0]}),
    };
    assert_eq!(deck.deck_id, DeckId(1234));
    assert_eq!(deck.name, "Japanese::Vocab");
    assert_eq!(deck.parent_name.as_deref(), Some("Japanese"));
}

#[test]
fn deck_without_parent() {
    let deck = AnkiDeck {
        deck_id: DeckId(1),
        name: "Default".into(),
        parent_name: None,
        config: serde_json::Value::Null,
    };
    assert!(deck.parent_name.is_none());
}

#[test]
fn deck_serialization_roundtrip() {
    let deck = AnkiDeck {
        deck_id: DeckId(42),
        name: "Test".into(),
        parent_name: None,
        config: serde_json::json!({}),
    };
    let json = serde_json::to_string(&deck).expect("serialize");
    let restored: AnkiDeck = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.deck_id, DeckId(42));
    assert_eq!(restored.name, "Test");
}

#[test]
fn deck_clone() {
    let deck = AnkiDeck {
        deck_id: DeckId(1),
        name: "A".into(),
        parent_name: None,
        config: serde_json::Value::Null,
    };
    let cloned = deck.clone();
    assert_eq!(cloned.deck_id, deck.deck_id);
    assert_eq!(cloned.name, deck.name);
}

// --- AnkiModel ---

#[test]
fn model_create_with_fields_and_templates() {
    let model = AnkiModel {
        model_id: ModelId(99),
        name: "Basic".into(),
        fields: vec![
            serde_json::json!({"name": "Front", "ord": 0}),
            serde_json::json!({"name": "Back", "ord": 1}),
        ],
        templates: vec![serde_json::json!({"name": "Card 1"})],
        config: serde_json::json!({}),
    };
    assert_eq!(model.model_id, ModelId(99));
    assert_eq!(model.name, "Basic");
    assert_eq!(model.fields.len(), 2);
    assert_eq!(model.templates.len(), 1);
}

#[test]
fn model_serialization_roundtrip() {
    let model = AnkiModel {
        model_id: ModelId(1),
        name: "Cloze".into(),
        fields: vec![],
        templates: vec![],
        config: serde_json::json!(null),
    };
    let json = serde_json::to_string(&model).expect("serialize");
    let restored: AnkiModel = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.name, "Cloze");
}

// --- AnkiNote ---

#[test]
fn note_create_with_fields_and_tags() {
    let mut fields_json = HashMap::new();
    fields_json.insert("Front".into(), "Hello".into());
    fields_json.insert("Back".into(), "World".into());

    let note = AnkiNote {
        note_id: NoteId(100),
        model_id: ModelId(1),
        tags: vec!["vocab".into(), "japanese".into()],
        fields: vec!["Hello".into(), "World".into()],
        fields_json,
        raw_fields: Some("Hello\x1fWorld".into()),
        normalized_text: String::new(),
        mtime: 1700000000,
        usn: -1,
    };
    assert_eq!(note.note_id, NoteId(100));
    assert_eq!(note.fields.len(), 2);
    assert_eq!(note.fields_json["Front"], "Hello");
    assert_eq!(note.tags, vec!["vocab", "japanese"]);
    assert_eq!(note.raw_fields.as_deref(), Some("Hello\x1fWorld"));
}

#[test]
fn note_empty_tags_and_fields() {
    let note = AnkiNote {
        note_id: NoteId(1),
        model_id: ModelId(1),
        tags: vec![],
        fields: vec![],
        fields_json: HashMap::new(),
        raw_fields: None,
        normalized_text: String::new(),
        mtime: 0,
        usn: 0,
    };
    assert!(note.tags.is_empty());
    assert!(note.fields.is_empty());
    assert!(note.raw_fields.is_none());
}

#[test]
fn note_serialization_roundtrip() {
    let note = AnkiNote {
        note_id: NoteId(42),
        model_id: ModelId(1),
        tags: vec!["tag1".into()],
        fields: vec!["f1".into()],
        fields_json: HashMap::new(),
        raw_fields: None,
        normalized_text: "some text".into(),
        mtime: 123,
        usn: 0,
    };
    let json = serde_json::to_string(&note).expect("serialize");
    let restored: AnkiNote = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.note_id, NoteId(42));
    assert_eq!(restored.normalized_text, "some text");
}

// --- AnkiCard ---

#[test]
fn card_create_with_scheduling_data() {
    let card = AnkiCard {
        card_id: CardId(500),
        note_id: NoteId(100),
        deck_id: DeckId(1),
        ord: 0,
        due: Some(1000),
        ivl: 30,
        ease: 2500,
        lapses: 2,
        reps: 10,
        queue: 2,
        card_type: 2,
        mtime: 1700000000,
        usn: -1,
    };
    assert_eq!(card.card_id, CardId(500));
    assert_eq!(card.ease, 2500);
    assert_eq!(card.queue, 2);
    assert_eq!(card.card_type, 2);
    assert_eq!(card.due, Some(1000));
}

#[test]
fn card_new_card_defaults() {
    let card = AnkiCard {
        card_id: CardId(1),
        note_id: NoteId(1),
        deck_id: DeckId(1),
        ord: 0,
        due: None,
        ivl: 0,
        ease: 0,
        lapses: 0,
        reps: 0,
        queue: 0,
        card_type: 0,
        mtime: 0,
        usn: 0,
    };
    assert_eq!(card.ivl, 0);
    assert_eq!(card.card_type, 0);
    assert!(card.due.is_none());
}

#[test]
fn card_suspended() {
    let card = AnkiCard {
        card_id: CardId(1),
        note_id: NoteId(1),
        deck_id: DeckId(1),
        ord: 0,
        due: None,
        ivl: 10,
        ease: 2500,
        lapses: 0,
        reps: 5,
        queue: -1, // suspended
        card_type: 2,
        mtime: 0,
        usn: 0,
    };
    assert_eq!(card.queue, -1);
}

#[test]
fn card_serialization_roundtrip() {
    let card = AnkiCard {
        card_id: CardId(1),
        note_id: NoteId(1),
        deck_id: DeckId(1),
        ord: 0,
        due: Some(5),
        ivl: 10,
        ease: 2500,
        lapses: 0,
        reps: 3,
        queue: 2,
        card_type: 2,
        mtime: 100,
        usn: 0,
    };
    let json = serde_json::to_string(&card).expect("serialize");
    let restored: AnkiCard = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.card_id, CardId(1));
    assert_eq!(restored.ease, 2500);
}

// --- AnkiRevlogEntry ---

#[test]
fn revlog_entry_create() {
    let entry = AnkiRevlogEntry {
        id: 1700000000000,
        card_id: CardId(500),
        usn: -1,
        button_chosen: 3,
        interval: 30,
        last_interval: 15,
        ease: 2500,
        time_ms: 8000,
        review_type: 1,
    };
    assert_eq!(entry.button_chosen, 3);
    assert_eq!(entry.time_ms, 8000);
    assert_eq!(entry.review_type, 1);
}

#[test]
fn revlog_fail_button() {
    let entry = AnkiRevlogEntry {
        id: 1,
        card_id: CardId(1),
        usn: 0,
        button_chosen: 1, // again/fail
        interval: 0,
        last_interval: 10,
        ease: 2100,
        time_ms: 15000,
        review_type: 2, // relearn
    };
    assert_eq!(entry.button_chosen, 1);
    assert_eq!(entry.review_type, 2);
}

#[test]
fn revlog_serialization_roundtrip() {
    let entry = AnkiRevlogEntry {
        id: 999,
        card_id: CardId(1),
        usn: 0,
        button_chosen: 4,
        interval: 60,
        last_interval: 30,
        ease: 2800,
        time_ms: 5000,
        review_type: 1,
    };
    let json = serde_json::to_string(&entry).expect("serialize");
    let restored: AnkiRevlogEntry = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.id, 999);
    assert_eq!(restored.button_chosen, 4);
}

// --- CardStats ---

#[test]
fn card_stats_with_review_data() {
    let now = Utc::now();
    let stats = CardStats {
        card_id: CardId(500),
        reviews: 15,
        avg_ease: Some(2450.0),
        fail_rate: Some(0.13),
        last_review_at: Some(now),
        total_time_ms: 120000,
    };
    assert_eq!(stats.reviews, 15);
    assert!((stats.avg_ease.unwrap() - 2450.0).abs() < f64::EPSILON);
    assert!(stats.last_review_at.is_some());
}

#[test]
fn card_stats_no_reviews() {
    let stats = CardStats {
        card_id: CardId(1),
        reviews: 0,
        avg_ease: None,
        fail_rate: None,
        last_review_at: None,
        total_time_ms: 0,
    };
    assert_eq!(stats.reviews, 0);
    assert!(stats.avg_ease.is_none());
    assert!(stats.fail_rate.is_none());
    assert!(stats.last_review_at.is_none());
}

#[test]
fn card_stats_serialization_roundtrip() {
    let stats = CardStats {
        card_id: CardId(1),
        reviews: 5,
        avg_ease: Some(2500.0),
        fail_rate: Some(0.2),
        last_review_at: None,
        total_time_ms: 50000,
    };
    let json = serde_json::to_string(&stats).expect("serialize");
    let restored: CardStats = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.reviews, 5);
    assert!((restored.fail_rate.unwrap() - 0.2).abs() < f64::EPSILON);
}

// --- AnkiCollection ---

#[test]
fn collection_default_schema_version() {
    let collection = AnkiCollection::default();
    assert_eq!(collection.schema_version, 11);
    assert!(collection.decks.is_empty());
    assert!(collection.models.is_empty());
    assert!(collection.notes.is_empty());
    assert!(collection.cards.is_empty());
    assert!(collection.card_stats.is_empty());
    assert!(collection.collection_path.is_none());
}

#[test]
fn collection_with_data() {
    let now = Utc::now();
    let collection = AnkiCollection {
        decks: vec![AnkiDeck {
            deck_id: DeckId(1),
            name: "Default".into(),
            parent_name: None,
            config: serde_json::Value::Null,
        }],
        models: vec![],
        notes: vec![],
        cards: vec![],
        card_stats: vec![],
        collection_path: Some("/path/to/collection.anki2".into()),
        extracted_at: now,
        schema_version: 11,
    };
    assert_eq!(collection.decks.len(), 1);
    assert_eq!(
        collection.collection_path.as_deref(),
        Some("/path/to/collection.anki2")
    );
}

#[test]
fn collection_serialization_roundtrip() {
    let collection = AnkiCollection::default();
    let json = serde_json::to_string(&collection).expect("serialize");
    let restored: AnkiCollection = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored.schema_version, 11);
}

// --- Send + Sync ---

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn all_types_are_send_and_sync() {
    assert_send::<AnkiDeck>();
    assert_sync::<AnkiDeck>();
    assert_send::<AnkiModel>();
    assert_sync::<AnkiModel>();
    assert_send::<AnkiNote>();
    assert_sync::<AnkiNote>();
    assert_send::<AnkiCard>();
    assert_sync::<AnkiCard>();
    assert_send::<AnkiRevlogEntry>();
    assert_sync::<AnkiRevlogEntry>();
    assert_send::<CardStats>();
    assert_sync::<CardStats>();
    assert_send::<AnkiCollection>();
    assert_sync::<AnkiCollection>();
}

// --- Debug ---

#[test]
fn all_types_implement_debug() {
    let deck = AnkiDeck {
        deck_id: DeckId(1),
        name: "Test".into(),
        parent_name: None,
        config: serde_json::Value::Null,
    };
    let _ = format!("{deck:?}");

    let note = AnkiNote {
        note_id: NoteId(1),
        model_id: ModelId(1),
        tags: vec![],
        fields: vec![],
        fields_json: HashMap::new(),
        raw_fields: None,
        normalized_text: String::new(),
        mtime: 0,
        usn: 0,
    };
    let _ = format!("{note:?}");

    let collection = AnkiCollection::default();
    let _ = format!("{collection:?}");
}

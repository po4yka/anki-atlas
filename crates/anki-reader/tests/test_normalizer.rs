use anki_reader::models::{AnkiCard, AnkiDeck, AnkiNote};
use anki_reader::normalizer::{
    build_card_deck_map, build_deck_map, classify_field, normalize_note, normalize_notes,
    normalize_whitespace, strip_html,
};
use common::{CardId, DeckId, ModelId, NoteId};
use std::collections::HashMap;

// --- strip_html ---

#[test]
fn strip_html_removes_bold() {
    assert_eq!(strip_html("<b>hello</b>", false), "hello");
}

#[test]
fn strip_html_removes_nested_tags() {
    assert_eq!(strip_html("<b><i>text</i></b>", false), "text");
}

#[test]
fn strip_html_br_to_newline() {
    let result = strip_html("line1<br>line2", false);
    assert!(result.contains("line1"));
    assert!(result.contains("line2"));
}

#[test]
fn strip_html_br_variants() {
    // <br/> and <br /> should also be handled
    let result = strip_html("a<br/>b<br />c", false);
    assert!(result.contains('a'));
    assert!(result.contains('b'));
    assert!(result.contains('c'));
}

#[test]
fn strip_html_nbsp() {
    let result = strip_html("hello&nbsp;world", false);
    assert!(result.contains("hello"));
    assert!(result.contains("world"));
}

#[test]
fn strip_html_entities() {
    let result = strip_html("&amp; &lt; &gt; &quot;", false);
    assert!(result.contains('&'));
    assert!(result.contains('<'));
    assert!(result.contains('>'));
    assert!(result.contains('"'));
}

#[test]
fn strip_html_empty_input() {
    assert_eq!(strip_html("", false), "");
}

#[test]
fn strip_html_plain_text_unchanged() {
    assert_eq!(strip_html("just plain text", false), "just plain text");
}

#[test]
fn strip_html_p_and_div_to_newlines() {
    let result = strip_html("<p>paragraph</p><div>block</div>", false);
    assert!(result.contains("paragraph"));
    assert!(result.contains("block"));
}

#[test]
fn strip_html_li_items() {
    let result = strip_html("<ul><li>item1</li><li>item2</li></ul>", false);
    assert!(result.contains("item1"));
    assert!(result.contains("item2"));
}

// --- strip_html: cloze deletions ---

#[test]
fn strip_html_cloze_simple() {
    assert_eq!(strip_html("{{c1::answer}}", false), "answer");
}

#[test]
fn strip_html_cloze_with_hint() {
    assert_eq!(strip_html("{{c1::answer::hint}}", false), "answer");
}

#[test]
fn strip_html_cloze_multiple() {
    let result = strip_html("{{c1::first}} and {{c2::second}}", false);
    assert!(result.contains("first"));
    assert!(result.contains("second"));
    assert!(!result.contains("c1"));
    assert!(!result.contains("c2"));
}

#[test]
fn strip_html_cloze_with_html() {
    let result = strip_html("<b>{{c1::bold answer}}</b>", false);
    assert_eq!(result, "bold answer");
}

// --- strip_html: preserve_code ---

#[test]
fn strip_html_preserve_code_inline() {
    let result = strip_html("use <code>fn main()</code> here", true);
    assert!(result.contains("`fn main()`"));
}

#[test]
fn strip_html_preserve_code_block() {
    let result = strip_html("<pre>let x = 1;\nlet y = 2;</pre>", true);
    assert!(result.contains('`'));
    assert!(result.contains("let x = 1;"));
}

#[test]
fn strip_html_no_preserve_code_strips_tags() {
    let result = strip_html("use <code>fn main()</code> here", false);
    assert!(result.contains("fn main()"));
    assert!(!result.contains("<code>"));
}

#[test]
fn strip_html_code_with_html_entities() {
    let result = strip_html("<code>&lt;div&gt;</code>", true);
    assert!(result.contains("`<div>`"));
}

// --- normalize_whitespace ---

#[test]
fn normalize_whitespace_collapses_spaces() {
    assert_eq!(normalize_whitespace("hello    world"), "hello world");
}

#[test]
fn normalize_whitespace_preserves_single_newlines() {
    let result = normalize_whitespace("line1\nline2");
    assert!(result.contains("line1"));
    assert!(result.contains("line2"));
}

#[test]
fn normalize_whitespace_removes_blank_lines() {
    let result: String = normalize_whitespace("line1\n\n\nline2");
    // Should not have empty lines between
    for line in result.lines() {
        assert!(!line.trim().is_empty());
    }
}

#[test]
fn normalize_whitespace_trims_lines() {
    let result: String = normalize_whitespace("  hello  \n  world  ");
    for line in result.lines() {
        assert_eq!(line, line.trim());
    }
}

#[test]
fn normalize_whitespace_empty() {
    assert_eq!(normalize_whitespace(""), "");
}

// --- classify_field ---

#[test]
fn classify_field_front_variants() {
    assert_eq!(classify_field("Front"), "front");
    assert_eq!(classify_field("front"), "front");
    assert_eq!(classify_field("Question"), "front");
    assert_eq!(classify_field("Expression"), "front");
    assert_eq!(classify_field("Word"), "front");
    assert_eq!(classify_field("Term"), "front");
    assert_eq!(classify_field("Prompt"), "front");
}

#[test]
fn classify_field_back_variants() {
    assert_eq!(classify_field("Back"), "back");
    assert_eq!(classify_field("back"), "back");
    assert_eq!(classify_field("Answer"), "back");
    assert_eq!(classify_field("Meaning"), "back");
    assert_eq!(classify_field("Definition"), "back");
    assert_eq!(classify_field("Response"), "back");
    assert_eq!(classify_field("Reading"), "back");
}

#[test]
fn classify_field_extra_variants() {
    assert_eq!(classify_field("Extra"), "extra");
    assert_eq!(classify_field("extra"), "extra");
    assert_eq!(classify_field("Notes"), "extra");
    assert_eq!(classify_field("Hint"), "extra");
    assert_eq!(classify_field("Example"), "extra");
    assert_eq!(classify_field("Examples"), "extra");
    assert_eq!(classify_field("Context"), "extra");
}

#[test]
fn classify_field_unknown() {
    assert_eq!(classify_field("xyz"), "other");
    assert_eq!(classify_field("Custom"), "other");
    assert_eq!(classify_field(""), "other");
}

#[test]
fn classify_field_case_insensitive() {
    assert_eq!(classify_field("FRONT"), "front");
    assert_eq!(classify_field("BACK"), "back");
    assert_eq!(classify_field("EXTRA"), "extra");
}

// --- normalize_note ---

fn make_note(fields_json: HashMap<String, String>, tags: Vec<String>) -> AnkiNote {
    AnkiNote {
        note_id: NoteId(1),
        model_id: ModelId(1),
        tags,
        fields: fields_json.values().cloned().collect(),
        fields_json,
        raw_fields: None,
        normalized_text: String::new(),
        mtime: 0,
        usn: 0,
    }
}

#[test]
fn normalize_note_basic() {
    let mut fields = HashMap::new();
    fields.insert("Front".into(), "Hello".into());
    fields.insert("Back".into(), "World".into());
    let note = make_note(fields, vec!["tag1".into()]);

    let result = normalize_note(&note, None);
    assert!(result.contains("Front:"));
    assert!(result.contains("Hello"));
    assert!(result.contains("Back:"));
    assert!(result.contains("World"));
    assert!(result.contains("Tags:"));
    assert!(result.contains("tag1"));
}

#[test]
fn normalize_note_with_decks() {
    let mut fields = HashMap::new();
    fields.insert("Front".into(), "Q".into());
    fields.insert("Back".into(), "A".into());
    let note = make_note(fields, vec![]);

    let deck_names = vec!["Japanese".to_string(), "Vocab".to_string()];
    let result = normalize_note(&note, Some(&deck_names));
    assert!(result.contains("Decks:"));
    assert!(result.contains("Japanese"));
    assert!(result.contains("Vocab"));
}

#[test]
fn normalize_note_strips_html_from_fields() {
    let mut fields = HashMap::new();
    fields.insert("Front".into(), "<b>Bold Question</b>".into());
    fields.insert("Back".into(), "<i>Italic Answer</i>".into());
    let note = make_note(fields, vec![]);

    let result = normalize_note(&note, None);
    assert!(result.contains("Bold Question"));
    assert!(result.contains("Italic Answer"));
    assert!(!result.contains("<b>"));
    assert!(!result.contains("<i>"));
}

#[test]
fn normalize_note_empty_fields() {
    let note = make_note(HashMap::new(), vec![]);
    let result = normalize_note(&note, None);
    // Should not panic, might be empty or minimal
    let _ = result;
}

#[test]
fn normalize_note_with_extra_field() {
    let mut fields = HashMap::new();
    fields.insert("Front".into(), "Q".into());
    fields.insert("Back".into(), "A".into());
    fields.insert("Extra".into(), "Some notes".into());
    let note = make_note(fields, vec![]);

    let result = normalize_note(&note, None);
    assert!(result.contains("Extra:"));
    assert!(result.contains("Some notes"));
}

// --- normalize_notes (batch) ---

#[test]
fn normalize_notes_populates_normalized_text() {
    let mut fields = HashMap::new();
    fields.insert("Front".into(), "Q".into());
    fields.insert("Back".into(), "A".into());
    let mut notes = vec![make_note(fields, vec!["tag".into()])];

    let deck_map: HashMap<DeckId, String> = HashMap::new();
    let card_deck_map: HashMap<NoteId, Vec<DeckId>> = HashMap::new();

    normalize_notes(&mut notes, &deck_map, &card_deck_map);
    assert!(!notes[0].normalized_text.is_empty());
}

// --- build_deck_map ---

#[test]
fn build_deck_map_creates_mapping() {
    let decks = vec![
        AnkiDeck {
            deck_id: DeckId(1),
            name: "Default".into(),
            parent_name: None,
            config: serde_json::Value::Null,
        },
        AnkiDeck {
            deck_id: DeckId(2),
            name: "Japanese".into(),
            parent_name: None,
            config: serde_json::Value::Null,
        },
    ];
    let map = build_deck_map(&decks);
    assert_eq!(map.len(), 2);
    assert_eq!(map[&DeckId(1)], "Default");
    assert_eq!(map[&DeckId(2)], "Japanese");
}

#[test]
fn build_deck_map_empty() {
    let map = build_deck_map(&[]);
    assert!(map.is_empty());
}

// --- build_card_deck_map ---

#[test]
fn build_card_deck_map_groups_by_note() {
    let cards = vec![
        AnkiCard {
            card_id: CardId(1),
            note_id: NoteId(100),
            deck_id: DeckId(10),
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
        },
        AnkiCard {
            card_id: CardId(2),
            note_id: NoteId(100),
            deck_id: DeckId(20),
            ord: 1,
            due: None,
            ivl: 0,
            ease: 0,
            lapses: 0,
            reps: 0,
            queue: 0,
            card_type: 0,
            mtime: 0,
            usn: 0,
        },
        AnkiCard {
            card_id: CardId(3),
            note_id: NoteId(200),
            deck_id: DeckId(10),
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
        },
    ];
    let map = build_card_deck_map(&cards);
    assert_eq!(map.len(), 2);
    assert!(map[&NoteId(100)].contains(&DeckId(10)));
    assert!(map[&NoteId(100)].contains(&DeckId(20)));
    assert_eq!(map[&NoteId(200)], vec![DeckId(10)]);
}

#[test]
fn build_card_deck_map_deduplicates() {
    let cards = vec![
        AnkiCard {
            card_id: CardId(1),
            note_id: NoteId(100),
            deck_id: DeckId(10),
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
        },
        AnkiCard {
            card_id: CardId(2),
            note_id: NoteId(100),
            deck_id: DeckId(10), // same deck as card 1
            ord: 1,
            due: None,
            ivl: 0,
            ease: 0,
            lapses: 0,
            reps: 0,
            queue: 0,
            card_type: 0,
            mtime: 0,
            usn: 0,
        },
    ];
    let map = build_card_deck_map(&cards);
    // deck_id 10 should appear only once for note 100
    assert_eq!(map[&NoteId(100)].len(), 1);
}

#[test]
fn build_card_deck_map_empty() {
    let map = build_card_deck_map(&[]);
    assert!(map.is_empty());
}

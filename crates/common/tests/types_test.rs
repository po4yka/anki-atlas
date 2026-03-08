use common::types::*;
use std::str::FromStr;

// ── Send + Sync ──────────────────────────────────────────────────────────

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn all_types_are_send_and_sync() {
    assert_send_sync::<Language>();
    assert_send_sync::<SlugStr>();
    assert_send_sync::<CardId>();
    assert_send_sync::<NoteId>();
    assert_send_sync::<DeckName>();
}

// ── Language enum ────────────────────────────────────────────────────────

#[test]
fn language_serde_roundtrip_lowercase() {
    // Spec: Language must serialize as lowercase "en", not "En"
    let lang = Language::En;
    let json = serde_json::to_string(&lang).unwrap();
    assert_eq!(json, r#""en""#, "Language::En should serialize as \"en\"");

    let parsed: Language = serde_json::from_str(r#""en""#).unwrap();
    assert_eq!(parsed, Language::En);
}

#[test]
fn language_all_variants_serde_roundtrip() {
    let cases = [
        (Language::En, "en"),
        (Language::Ru, "ru"),
        (Language::De, "de"),
        (Language::Fr, "fr"),
        (Language::Es, "es"),
        (Language::It, "it"),
        (Language::Pt, "pt"),
        (Language::Zh, "zh"),
        (Language::Ja, "ja"),
        (Language::Ko, "ko"),
    ];

    for (variant, expected_str) in cases {
        let json = serde_json::to_string(&variant).unwrap();
        let expected_json = format!(r#""{expected_str}""#);
        assert_eq!(json, expected_json, "Serialization mismatch for {variant:?}");

        let deserialized: Language = serde_json::from_str(&expected_json).unwrap();
        assert_eq!(deserialized, variant, "Deserialization mismatch for {expected_str}");
    }
}

#[test]
fn language_display_lowercase() {
    assert_eq!(Language::En.to_string(), "en");
    assert_eq!(Language::Ru.to_string(), "ru");
    assert_eq!(Language::Zh.to_string(), "zh");
}

#[test]
fn language_from_str_lowercase() {
    assert_eq!(Language::from_str("en").unwrap(), Language::En);
    assert_eq!(Language::from_str("ru").unwrap(), Language::Ru);
    assert_eq!(Language::from_str("ko").unwrap(), Language::Ko);
}

#[test]
fn language_from_str_invalid() {
    assert!(Language::from_str("xx").is_err());
    assert!(Language::from_str("").is_err());
    assert!(Language::from_str("english").is_err());
}

#[test]
fn language_clone_copy() {
    let lang = Language::En;
    let cloned = lang.clone();
    let copied = lang;
    assert_eq!(lang, cloned);
    assert_eq!(lang, copied);
}

#[test]
fn language_hash_eq() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(Language::En);
    set.insert(Language::En);
    assert_eq!(set.len(), 1);
    set.insert(Language::Ru);
    assert_eq!(set.len(), 2);
}

// ── SlugStr ──────────────────────────────────────────────────────────────

#[test]
fn slug_str_serde_roundtrip() {
    let slug = SlugStr("kotlin-coroutines-basics".to_string());
    let json = serde_json::to_string(&slug).unwrap();
    assert_eq!(json, r#""kotlin-coroutines-basics""#);

    let parsed: SlugStr = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, slug);
}

#[test]
fn slug_str_empty() {
    let slug = SlugStr(String::new());
    let json = serde_json::to_string(&slug).unwrap();
    assert_eq!(json, r#""""#);

    let parsed: SlugStr = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.0, "");
}

#[test]
fn slug_str_hash_eq() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(SlugStr("a".to_string()));
    set.insert(SlugStr("a".to_string()));
    assert_eq!(set.len(), 1);
}

// ── CardId ───────────────────────────────────────────────────────────────

#[test]
fn card_id_serde_roundtrip() {
    let id = CardId(42);
    let json = serde_json::to_string(&id).unwrap();
    assert_eq!(json, "42");

    let parsed: CardId = serde_json::from_str("42").unwrap();
    assert_eq!(parsed, id);
}

#[test]
fn card_id_zero_and_negative() {
    let zero = CardId(0);
    let json = serde_json::to_string(&zero).unwrap();
    assert_eq!(json, "0");

    let neg = CardId(-1);
    let json = serde_json::to_string(&neg).unwrap();
    assert_eq!(json, "-1");
}

#[test]
fn card_id_copy() {
    let id = CardId(1);
    let copied = id;
    assert_eq!(id, copied);
}

// ── NoteId ───────────────────────────────────────────────────────────────

#[test]
fn note_id_serde_roundtrip() {
    let id = NoteId(999);
    let json = serde_json::to_string(&id).unwrap();
    assert_eq!(json, "999");

    let parsed: NoteId = serde_json::from_str("999").unwrap();
    assert_eq!(parsed, id);
}

#[test]
fn note_id_large_value() {
    let id = NoteId(i64::MAX);
    let json = serde_json::to_string(&id).unwrap();
    let parsed: NoteId = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, id);
}

// ── DeckName ─────────────────────────────────────────────────────────────

#[test]
fn deck_name_serde_roundtrip() {
    let deck = DeckName("Parent::Child::Grandchild".to_string());
    let json = serde_json::to_string(&deck).unwrap();
    assert_eq!(json, r#""Parent::Child::Grandchild""#);

    let parsed: DeckName = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, deck);
}

#[test]
fn deck_name_with_hierarchy_separator() {
    let deck = DeckName("Languages::Rust::Ownership".to_string());
    assert!(deck.0.contains("::"));
}

#[test]
fn deck_name_hash_eq() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(DeckName("A".to_string()));
    set.insert(DeckName("A".to_string()));
    assert_eq!(set.len(), 1);
}

// ── Debug impls ──────────────────────────────────────────────────────────

#[test]
fn all_types_implement_debug() {
    let _ = format!("{:?}", Language::En);
    let _ = format!("{:?}", SlugStr("test".to_string()));
    let _ = format!("{:?}", CardId(1));
    let _ = format!("{:?}", NoteId(1));
    let _ = format!("{:?}", DeckName("D".to_string()));
}

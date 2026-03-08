use crate::models::{AnkiCard, AnkiDeck, AnkiNote};
use std::collections::HashMap;

/// Strip HTML tags from text.
pub fn strip_html(_text: &str, _preserve_code: bool) -> String {
    todo!()
}

/// Normalize whitespace.
pub fn normalize_whitespace(_text: &str) -> String {
    todo!()
}

/// Classify a field name as "front", "back", "extra", or "other".
pub fn classify_field(_name: &str) -> &'static str {
    todo!()
}

/// Normalize a single note to searchable text.
pub fn normalize_note(_note: &AnkiNote, _deck_names: Option<&[String]>) -> String {
    todo!()
}

/// Normalize all notes in-place.
pub fn normalize_notes(
    _notes: &mut [AnkiNote],
    _deck_map: &HashMap<i64, String>,
    _card_deck_map: &HashMap<i64, Vec<i64>>,
) {
    todo!()
}

/// Build deck_id -> deck_name mapping.
pub fn build_deck_map(_decks: &[AnkiDeck]) -> HashMap<i64, String> {
    todo!()
}

/// Build note_id -> vec of deck_ids mapping from cards.
pub fn build_card_deck_map(_cards: &[AnkiCard]) -> HashMap<i64, Vec<i64>> {
    todo!()
}

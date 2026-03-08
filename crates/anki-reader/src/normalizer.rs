use crate::models::{AnkiCard, AnkiDeck, AnkiNote};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

static CLOZE_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\{\{c\d+::([^}]+?)(?:::[^}]+)?\}\}").unwrap());

static HTML_TAG_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]+>").unwrap());

static WHITESPACE_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

static CODE_BLOCK_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<pre[^>]*>(.*?)</pre>").unwrap());

static CODE_INLINE_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<code[^>]*>(.*?)</code>").unwrap());

/// Strip HTML tags from text.
pub fn strip_html(text: &str, preserve_code: bool) -> String {
    if text.is_empty() {
        return String::new();
    }

    let mut result = text.to_string();
    let mut code_blocks: Vec<String> = Vec::new();

    // 1. Save code blocks as placeholders
    if preserve_code {
        let mut idx = 0usize;
        for pat in [&*CODE_BLOCK_PATTERN, &*CODE_INLINE_PATTERN] {
            while let Some(cap) = pat.captures(&result) {
                let full_match = cap.get(0).unwrap().as_str().to_string();
                let content = cap.get(1).unwrap().as_str().to_string();
                let placeholder = format!("__CODE_BLOCK_{idx}__");
                code_blocks.push(content);
                result = result.replacen(&full_match, &placeholder, 1);
                idx += 1;
            }
        }
    }

    // 2. Expand cloze deletions
    result = CLOZE_PATTERN.replace_all(&result, "$1").to_string();

    // 3. Replace block-level tags with newlines
    result = result.replace("<br>", "\n");
    result = result.replace("<br/>", "\n");
    result = result.replace("<br />", "\n");
    result = result.replace("&nbsp;", " ");

    for tag in ["<p>", "</p>", "<div>", "</div>"] {
        result = result.replace(tag, "\n");
    }
    // Case-insensitive variants
    for tag in ["<P>", "</P>", "<DIV>", "</DIV>"] {
        result = result.replace(tag, "\n");
    }
    result = result.replace("<li>", "\n");
    result = result.replace("<LI>", "\n");
    result = result.replace("</li>", "");
    result = result.replace("</LI>", "");

    // 4. Strip remaining HTML tags
    result = HTML_TAG_PATTERN.replace_all(&result, "").to_string();

    // 5. Decode HTML entities
    result = html_unescape(&result);

    // 6. Restore code blocks wrapped in backticks
    if preserve_code {
        for (i, block) in code_blocks.iter().enumerate() {
            let placeholder = format!("__CODE_BLOCK_{i}__");
            let decoded = html_unescape(block);
            result = result.replace(&placeholder, &format!("`{decoded}`"));
        }
    }

    // 7. Collapse whitespace
    result = WHITESPACE_PATTERN.replace_all(&result, " ").to_string();

    result.trim().to_string()
}

fn html_unescape(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&#x27;", "'")
        .replace("&nbsp;", " ")
}

/// Normalize whitespace: collapse runs, preserve single newlines, trim lines, remove blank lines.
pub fn normalize_whitespace(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    text.lines()
        .map(|line| {
            WHITESPACE_PATTERN
                .replace_all(line.trim(), " ")
                .to_string()
        })
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Classify a field name as "front", "back", "extra", or "other".
pub fn classify_field(name: &str) -> &'static str {
    let lower = name.to_lowercase();
    match lower.as_str() {
        "front" | "question" | "expression" | "word" | "term" | "prompt" => "front",
        "back" | "answer" | "meaning" | "definition" | "response" | "reading" => "back",
        "extra" | "notes" | "hint" | "example" | "examples" | "context" => "extra",
        _ => "other",
    }
}

/// Normalize a single note to searchable text.
pub fn normalize_note(note: &AnkiNote, deck_names: Option<&[String]>) -> String {
    let mut front_parts = Vec::new();
    let mut back_parts = Vec::new();
    let mut extra_parts = Vec::new();
    let mut other_parts = Vec::new();
    let mut has_classified = false;

    for (name, value) in &note.fields_json {
        let cleaned = strip_html(value, false);
        if cleaned.is_empty() {
            continue;
        }
        match classify_field(name) {
            "front" => {
                front_parts.push(cleaned);
                has_classified = true;
            }
            "back" => {
                back_parts.push(cleaned);
                has_classified = true;
            }
            "extra" => {
                extra_parts.push(cleaned);
                has_classified = true;
            }
            _ => other_parts.push(cleaned),
        }
    }

    // Fallback: if no fields matched front/back/extra, use positional
    if !has_classified && !note.fields.is_empty() {
        for (i, value) in note.fields.iter().enumerate() {
            let cleaned = strip_html(value, false);
            if cleaned.is_empty() {
                continue;
            }
            match i {
                0 => front_parts.push(cleaned),
                1 => back_parts.push(cleaned),
                _ => extra_parts.push(cleaned),
            }
        }
    }

    let mut lines = Vec::new();

    if !front_parts.is_empty() {
        lines.push(format!("Front: {}", front_parts.join(" ")));
    }
    if !back_parts.is_empty() {
        lines.push(format!("Back: {}", back_parts.join(" ")));
    }
    if !extra_parts.is_empty() {
        lines.push(format!("Extra: {}", extra_parts.join(" ")));
    }
    for part in &other_parts {
        lines.push(part.clone());
    }
    if !note.tags.is_empty() {
        lines.push(format!("Tags: {}", note.tags.join(", ")));
    }
    if let Some(decks) = deck_names {
        if !decks.is_empty() {
            lines.push(format!("Decks: {}", decks.join(", ")));
        }
    }

    normalize_whitespace(&lines.join("\n"))
}

/// Normalize all notes in-place, populating `normalized_text`.
pub fn normalize_notes(
    notes: &mut [AnkiNote],
    deck_map: &HashMap<i64, String>,
    card_deck_map: &HashMap<i64, Vec<i64>>,
) {
    for note in notes.iter_mut() {
        let deck_names: Option<Vec<String>> = card_deck_map.get(&note.note_id).map(|deck_ids| {
            deck_ids
                .iter()
                .filter_map(|id| deck_map.get(id).cloned())
                .collect()
        });
        note.normalized_text = normalize_note(note, deck_names.as_deref());
    }
}

/// Build deck_id -> deck_name mapping.
pub fn build_deck_map(decks: &[AnkiDeck]) -> HashMap<i64, String> {
    decks.iter().map(|d| (d.deck_id, d.name.clone())).collect()
}

/// Build note_id -> vec of deck_ids mapping from cards.
pub fn build_card_deck_map(cards: &[AnkiCard]) -> HashMap<i64, Vec<i64>> {
    let mut map: HashMap<i64, Vec<i64>> = HashMap::new();
    for card in cards {
        let entry = map.entry(card.note_id).or_default();
        if !entry.contains(&card.deck_id) {
            entry.push(card.deck_id);
        }
    }
    map
}

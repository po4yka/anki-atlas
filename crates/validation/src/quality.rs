//! Heuristic quality scoring for flashcards.
//!
//! Implements a 5-dimension rubric: clarity, atomicity, testability,
//! memorability, accuracy. Each dimension scores 0.0-1.0.

use serde::Serialize;

/// Five-dimension quality assessment of a flashcard.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QualityScore {
    pub clarity: f64,
    pub atomicity: f64,
    pub testability: f64,
    pub memorability: f64,
    pub accuracy: f64,
}

impl QualityScore {
    /// Average of all five dimensions.
    pub fn overall(&self) -> f64 {
        (self.clarity + self.atomicity + self.testability + self.memorability + self.accuracy) / 5.0
    }
}

/// Score a card using heuristic checks. Each dimension is 0.0-1.0.
///
/// Dimensions:
/// - clarity: penalizes vague openers, yes/no questions, missing "?"
/// - atomicity: penalizes long questions (>20 words), multi-concept splits
/// - testability: penalizes extremely long answers (>100 words), empty answers
/// - memorability: penalizes long enumerations (>4 bullet items), long answers (>150 words)
/// - accuracy: penalizes missing question mark, empty front/back
pub fn assess_quality(front: &str, back: &str) -> QualityScore {
    let front_trimmed = front.trim();
    let back_trimmed = back.trim();

    QualityScore {
        clarity: score_clarity(front_trimmed),
        atomicity: score_atomicity(front_trimmed),
        testability: score_testability(back_trimmed),
        memorability: score_memorability(back_trimmed),
        accuracy: score_accuracy(front_trimmed, back_trimmed),
    }
}

fn score_clarity(front: &str) -> f64 {
    let mut score = 1.0;
    let lower = front.to_lowercase();

    let vague_patterns = [
        "explain",
        "describe",
        "tell me about",
        "tell about",
        "discuss",
        "elaborate",
    ];
    if vague_patterns.iter().any(|p| lower.contains(p)) {
        score -= 0.4;
    }

    let yes_no_starters = [
        "is ", "does ", "can ", "do ", "are ", "was ", "were ", "has ", "have ", "will ", "would ",
        "should ", "could ", "did ",
    ];
    if yes_no_starters.iter().any(|s| lower.starts_with(s)) {
        score -= 0.3;
    }

    if !front.is_empty() && !front.contains('?') {
        score -= 0.2;
    }

    clamp(score)
}

fn score_atomicity(front: &str) -> f64 {
    let mut score = 1.0;
    let word_count = front.split_whitespace().count();

    if word_count > 30 {
        score -= 0.4;
    } else if word_count > 20 {
        score -= 0.2;
    }

    let lower = front.to_lowercase();
    let and_or_count = count_word_boundary(&lower, "and") + count_word_boundary(&lower, "or");

    if and_or_count >= 2 {
        score -= 0.4;
    } else if and_or_count == 1 {
        score -= 0.1;
    }

    clamp(score)
}

fn count_word_boundary(text: &str, word: &str) -> usize {
    let mut count = 0;
    let chars: Vec<char> = text.chars().collect();
    let word_chars: Vec<char> = word.chars().collect();
    let wlen = word_chars.len();

    if chars.len() < wlen {
        return 0;
    }

    for i in 0..=chars.len() - wlen {
        let before_ok = i == 0 || !chars[i - 1].is_alphanumeric();
        let after_ok = i + wlen == chars.len() || !chars[i + wlen].is_alphanumeric();
        let matches = chars[i..i + wlen] == word_chars[..];

        if before_ok && after_ok && matches {
            count += 1;
        }
    }
    count
}

fn score_testability(back: &str) -> f64 {
    if back.is_empty() {
        return 0.0;
    }

    let mut score = 1.0;
    let word_count = back.split_whitespace().count();

    if word_count > 200 {
        score -= 0.5;
    } else if word_count > 100 {
        score -= 0.3;
    }

    clamp(score)
}

fn score_memorability(back: &str) -> f64 {
    let mut score = 1.0;

    let bullet_count = back
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            trimmed.starts_with("- ") || trimmed.starts_with("* ") || is_numbered_item(trimmed)
        })
        .count();

    if bullet_count > 7 {
        score -= 0.5;
    } else if bullet_count > 4 {
        score -= 0.2;
    }

    let word_count = back.split_whitespace().count();
    if word_count > 150 {
        score -= 0.3;
    }

    clamp(score)
}

fn is_numbered_item(line: &str) -> bool {
    let mut chars = line.chars().peekable();

    // Must start with a digit
    if !chars.peek().is_some_and(|c| c.is_ascii_digit()) {
        return false;
    }

    // Consume digits
    while chars.peek().is_some_and(|c| c.is_ascii_digit()) {
        chars.next();
    }

    // Must be followed by ". " or ") "
    match chars.next() {
        Some('.') | Some(')') => chars.next() == Some(' '),
        _ => false,
    }
}

fn score_accuracy(front: &str, back: &str) -> f64 {
    let mut score = 1.0;

    if front.is_empty() {
        score -= 0.5;
    }
    if back.is_empty() {
        score -= 0.5;
    }

    if !front.is_empty() && !front.contains('?') {
        score -= 0.2;
    }

    clamp(score)
}

fn clamp(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

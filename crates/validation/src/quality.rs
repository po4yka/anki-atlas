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
        todo!()
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
    todo!()
}

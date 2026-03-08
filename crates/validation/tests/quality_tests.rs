use validation::{assess_quality, QualityScore};

// ── QualityScore::overall ──────────────────────────────────────────

#[test]
fn overall_is_mean_of_five_dimensions() {
    let score = QualityScore {
        clarity: 1.0,
        atomicity: 0.8,
        testability: 0.6,
        memorability: 0.4,
        accuracy: 0.2,
    };
    let expected = (1.0 + 0.8 + 0.6 + 0.4 + 0.2) / 5.0;
    assert!((score.overall() - expected).abs() < f64::EPSILON);
}

#[test]
fn overall_perfect_score() {
    let score = QualityScore {
        clarity: 1.0,
        atomicity: 1.0,
        testability: 1.0,
        memorability: 1.0,
        accuracy: 1.0,
    };
    assert!((score.overall() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn overall_all_zero() {
    let score = QualityScore {
        clarity: 0.0,
        atomicity: 0.0,
        testability: 0.0,
        memorability: 0.0,
        accuracy: 0.0,
    };
    assert!((score.overall() - 0.0).abs() < f64::EPSILON);
}

// ── Well-formed card ───────────────────────────────────────────────

#[test]
fn well_formed_card_scores_all_ones() {
    let score = assess_quality("What is polymorphism?", "The ability of objects to take many forms.");
    assert!((score.overall() - 1.0).abs() < f64::EPSILON);
    assert!((score.clarity - 1.0).abs() < f64::EPSILON);
    assert!((score.atomicity - 1.0).abs() < f64::EPSILON);
    assert!((score.testability - 1.0).abs() < f64::EPSILON);
    assert!((score.memorability - 1.0).abs() < f64::EPSILON);
    assert!((score.accuracy - 1.0).abs() < f64::EPSILON);
}

// ── Clarity ────────────────────────────────────────────────────────

#[test]
fn clarity_penalizes_explain() {
    let score = assess_quality("Explain how TCP works?", "It works via handshake.");
    assert!(score.clarity < 1.0, "explain should penalize clarity");
    assert!((score.clarity - 0.6).abs() < f64::EPSILON, "clarity should be 1.0 - 0.4 = 0.6");
}

#[test]
fn clarity_penalizes_describe() {
    let score = assess_quality("Describe the process?", "Step by step.");
    assert!(score.clarity < 1.0);
}

#[test]
fn clarity_penalizes_tell_me_about() {
    let score = assess_quality("Tell me about Rust?", "It is a systems language.");
    assert!(score.clarity < 1.0);
}

#[test]
fn clarity_penalizes_tell_about() {
    let score = assess_quality("Tell about Rust?", "It is a systems language.");
    assert!(score.clarity < 1.0);
}

#[test]
fn clarity_penalizes_discuss() {
    let score = assess_quality("Discuss the tradeoffs?", "There are many.");
    assert!(score.clarity < 1.0);
}

#[test]
fn clarity_penalizes_elaborate() {
    let score = assess_quality("Elaborate on ownership?", "It is about memory.");
    assert!(score.clarity < 1.0);
}

#[test]
fn clarity_penalizes_elaborate_without_on() {
    let score = assess_quality("Elaborate the concept?", "Details here.");
    assert!(score.clarity < 1.0);
}

#[test]
fn clarity_vague_is_case_insensitive() {
    let score = assess_quality("EXPLAIN how it works?", "It works.");
    assert!(score.clarity < 1.0);
}

#[test]
fn clarity_penalizes_yes_no_starter_is() {
    let score = assess_quality("Is Rust memory safe?", "Yes.");
    assert!(score.clarity < 1.0, "yes/no starter should penalize clarity");
    // 1.0 - 0.3 (yes/no) = 0.7
    assert!((score.clarity - 0.7).abs() < f64::EPSILON);
}

#[test]
fn clarity_penalizes_yes_no_starter_does() {
    let score = assess_quality("Does it compile?", "Yes.");
    assert!(score.clarity < 1.0);
}

#[test]
fn clarity_penalizes_yes_no_starter_can() {
    let score = assess_quality("Can you do this?", "Yes.");
    assert!(score.clarity < 1.0);
}

#[test]
fn clarity_yes_no_is_case_insensitive() {
    let score = assess_quality("Is Rust fast?", "Yes.");
    // "is " after lowering
    assert!(score.clarity < 1.0);
}

#[test]
fn clarity_penalizes_missing_question_mark() {
    let score = assess_quality("What is Rust", "A language.");
    assert!(score.clarity < 1.0, "missing ? should penalize clarity");
    // 1.0 - 0.2 = 0.8
    assert!((score.clarity - 0.8).abs() < f64::EPSILON);
}

#[test]
fn clarity_cumulative_vague_and_no_question_mark() {
    // "explain" (-0.4) + no "?" (-0.2) = 0.4
    let score = assess_quality("Explain ownership", "Memory management.");
    assert!((score.clarity - 0.4).abs() < f64::EPSILON);
}

#[test]
fn clarity_cumulative_yes_no_and_no_question_mark() {
    // "is " (-0.3) + no "?" (-0.2) = 0.5
    let score = assess_quality("Is Rust fast", "Yes.");
    assert!((score.clarity - 0.5).abs() < f64::EPSILON);
}

#[test]
fn clarity_floor_at_zero() {
    // "explain" (-0.4) + "is " after stripping... actually explain doesn't start with yes/no
    // Let's try a combo that goes below zero: vague (-0.4) + yes/no (-0.3) + no ? (-0.2) = 0.1
    // "explain" alone won't start with yes/no. Let's do yes/no + no ?
    // Actually: explain (-0.4) + no ? (-0.2) = 0.4, that's still positive.
    // We need all three: but explain + yes/no won't both match on same string.
    // Floor is already tested by cumulative. Let's just verify floor doesn't go negative.
    let score = assess_quality("Explain ownership", "Details.");
    assert!(score.clarity >= 0.0);
}

// ── Atomicity ──────────────────────────────────────────────────────

#[test]
fn atomicity_perfect_for_short_question() {
    let score = assess_quality("What is Rust?", "A language.");
    assert!((score.atomicity - 1.0).abs() < f64::EPSILON);
}

#[test]
fn atomicity_penalizes_long_question_over_20_words() {
    let words: Vec<&str> = std::iter::repeat("word").take(22).collect();
    let front = format!("{}?", words.join(" "));
    let score = assess_quality(&front, "Answer.");
    assert!(score.atomicity < 1.0, ">20 words should penalize atomicity");
    // -0.2 for >20 words
    assert!((score.atomicity - 0.8).abs() < f64::EPSILON);
}

#[test]
fn atomicity_penalizes_very_long_question_over_30_words() {
    let words: Vec<&str> = std::iter::repeat("word").take(32).collect();
    let front = format!("{}?", words.join(" "));
    let score = assess_quality(&front, "Answer.");
    // -0.4 for >30 words
    assert!((score.atomicity - 0.6).abs() < f64::EPSILON);
}

#[test]
fn atomicity_penalizes_single_and_or() {
    let score = assess_quality("What is ownership and borrowing?", "Concepts.");
    // 1 occurrence of "and" -> -0.1
    assert!((score.atomicity - 0.9).abs() < f64::EPSILON);
}

#[test]
fn atomicity_penalizes_multiple_and_or() {
    let score = assess_quality("What is ownership and borrowing and lifetimes?", "Concepts.");
    // 2+ occurrences -> -0.4
    assert!((score.atomicity - 0.6).abs() < f64::EPSILON);
}

#[test]
fn atomicity_penalizes_or_connector() {
    let score = assess_quality("Is it stack or heap?", "Stack.");
    // 1 occurrence of "or" -> -0.1
    assert!(score.atomicity < 1.0);
}

#[test]
fn atomicity_floor_at_zero() {
    // >30 words (-0.4) + 2+ and/or (-0.4) = 0.2, still positive
    // Make it even worse: lots of and/or in a long sentence
    let front = "word and word and word and word word word word word word word word word word word word word word word word word word word word word word word word word word word word?";
    let score = assess_quality(front, "Answer.");
    assert!(score.atomicity >= 0.0);
}

// ── Testability ────────────────────────────────────────────────────

#[test]
fn testability_perfect_for_concise_answer() {
    let score = assess_quality("What is Rust?", "A systems programming language.");
    assert!((score.testability - 1.0).abs() < f64::EPSILON);
}

#[test]
fn testability_zero_for_empty_back() {
    let score = assess_quality("What is Rust?", "");
    assert!((score.testability - 0.0).abs() < f64::EPSILON);
}

#[test]
fn testability_zero_for_whitespace_only_back() {
    let score = assess_quality("What is Rust?", "   ");
    assert!((score.testability - 0.0).abs() < f64::EPSILON);
}

#[test]
fn testability_penalizes_long_answer_over_100_words() {
    let words: Vec<&str> = std::iter::repeat("word").take(105).collect();
    let back = words.join(" ");
    let score = assess_quality("What is Rust?", &back);
    // -0.3 for >100 words
    assert!((score.testability - 0.7).abs() < f64::EPSILON);
}

#[test]
fn testability_penalizes_very_long_answer_over_200_words() {
    let words: Vec<&str> = std::iter::repeat("word").take(205).collect();
    let back = words.join(" ");
    let score = assess_quality("What is Rust?", &back);
    // -0.5 for >200 words
    assert!((score.testability - 0.5).abs() < f64::EPSILON);
}

// ── Memorability ───────────────────────────────────────────────────

#[test]
fn memorability_perfect_for_short_answer() {
    let score = assess_quality("What is Rust?", "A language for safe systems.");
    assert!((score.memorability - 1.0).abs() < f64::EPSILON);
}

#[test]
fn memorability_penalizes_many_bullet_items_over_4() {
    let back = "- item1\n- item2\n- item3\n- item4\n- item5";
    let score = assess_quality("What are the features?", back);
    // 5 items > 4 -> -0.2
    assert!((score.memorability - 0.8).abs() < f64::EPSILON);
}

#[test]
fn memorability_penalizes_many_bullet_items_over_7() {
    let back = "- a\n- b\n- c\n- d\n- e\n- f\n- g\n- h";
    let score = assess_quality("What are the features?", back);
    // 8 items > 7 -> -0.5
    assert!((score.memorability - 0.5).abs() < f64::EPSILON);
}

#[test]
fn memorability_counts_numbered_items() {
    let back = "1. first\n2. second\n3. third\n4. fourth\n5. fifth";
    let score = assess_quality("What are the steps?", back);
    // 5 items > 4 -> -0.2
    assert!((score.memorability - 0.8).abs() < f64::EPSILON);
}

#[test]
fn memorability_counts_asterisk_items() {
    let back = "* a\n* b\n* c\n* d\n* e";
    let score = assess_quality("What are them?", back);
    // 5 items > 4 -> -0.2
    assert!(score.memorability < 1.0);
}

#[test]
fn memorability_penalizes_long_answer_over_150_words() {
    let words: Vec<&str> = std::iter::repeat("word").take(155).collect();
    let back = words.join(" ");
    let score = assess_quality("What is Rust?", &back);
    // -0.3 for >150 words
    assert!((score.memorability - 0.7).abs() < f64::EPSILON);
}

#[test]
fn memorability_cumulative_bullets_and_length() {
    // >7 bullets (-0.5) + >150 words (-0.3) = 0.2
    let mut lines: Vec<String> = (0..8).map(|i| format!("- item {i} with some extra words to pad it out more")).collect();
    // Add more filler words to push over 150
    let filler: Vec<&str> = std::iter::repeat("filler").take(120).collect();
    lines.push(filler.join(" "));
    let back = lines.join("\n");
    let score = assess_quality("What are the features?", &back);
    assert!((score.memorability - 0.2).abs() < f64::EPSILON);
}

#[test]
fn memorability_floor_at_zero() {
    let score = assess_quality("What?", "- a\n- b\n- c\n- d\n- e\n- f\n- g\n- h");
    assert!(score.memorability >= 0.0);
}

// ── Accuracy ───────────────────────────────────────────────────────

#[test]
fn accuracy_perfect_for_well_formed() {
    let score = assess_quality("What is Rust?", "A language.");
    assert!((score.accuracy - 1.0).abs() < f64::EPSILON);
}

#[test]
fn accuracy_penalizes_empty_front() {
    let score = assess_quality("", "Some answer.");
    assert!(score.accuracy < 1.0);
    // -0.5 for empty front, -0.2 for no ? (but front is empty, so no ? penalty only if front is non-empty)
    // From Python: if not front: -0.5, then "if front and '?' not in front" so no ? penalty when empty
    assert!((score.accuracy - 0.5).abs() < f64::EPSILON);
}

#[test]
fn accuracy_penalizes_empty_back() {
    let score = assess_quality("What is Rust?", "");
    // -0.5 for empty back
    assert!((score.accuracy - 0.5).abs() < f64::EPSILON);
}

#[test]
fn accuracy_penalizes_both_empty() {
    let score = assess_quality("", "");
    // -0.5 front + -0.5 back = 0.0
    assert!((score.accuracy - 0.0).abs() < f64::EPSILON);
}

#[test]
fn accuracy_penalizes_missing_question_mark() {
    let score = assess_quality("What is Rust", "A language.");
    // -0.2 for no ?
    assert!((score.accuracy - 0.8).abs() < f64::EPSILON);
}

#[test]
fn accuracy_no_question_mark_penalty_when_front_empty() {
    // Python: "if front and '?' not in front" - no penalty when front is empty
    let score = assess_quality("", "Answer.");
    // Only -0.5 for empty front, no -0.2 for missing ?
    assert!((score.accuracy - 0.5).abs() < f64::EPSILON);
}

#[test]
fn accuracy_floor_at_zero() {
    let score = assess_quality("", "");
    assert!(score.accuracy >= 0.0);
}

// ── Edge cases ─────────────────────────────────────────────────────

#[test]
fn all_dimensions_clamped_to_zero_one() {
    let score = assess_quality("", "");
    assert!(score.clarity >= 0.0 && score.clarity <= 1.0);
    assert!(score.atomicity >= 0.0 && score.atomicity <= 1.0);
    assert!(score.testability >= 0.0 && score.testability <= 1.0);
    assert!(score.memorability >= 0.0 && score.memorability <= 1.0);
    assert!(score.accuracy >= 0.0 && score.accuracy <= 1.0);
}

#[test]
fn whitespace_only_front_treated_as_empty() {
    let score = assess_quality("   \t\n  ", "Answer.");
    // After strip, front is empty -> accuracy -0.5
    assert!(score.accuracy < 1.0);
}

#[test]
fn quality_score_is_serializable() {
    let score = QualityScore {
        clarity: 1.0,
        atomicity: 0.5,
        testability: 0.8,
        memorability: 0.6,
        accuracy: 0.9,
    };
    let json = serde_json::to_string(&score).unwrap();
    assert!(json.contains("clarity"));
    assert!(json.contains("atomicity"));
}

#[test]
fn quality_score_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<QualityScore>();
}

// ── Numbered enumerations with closing paren ───────────────────────

#[test]
fn memorability_counts_numbered_with_closing_paren() {
    let back = "1) first\n2) second\n3) third\n4) fourth\n5) fifth";
    let score = assess_quality("What are them?", back);
    // 5 items > 4 -> -0.2
    assert!(score.memorability < 1.0);
}

// ── Atomicity: "or" as word boundary ───────────────────────────────

#[test]
fn atomicity_does_not_penalize_or_inside_words() {
    // "order" contains "or" but not as a word boundary
    let score = assess_quality("What is the order of operations?", "PEMDAS.");
    // No word-boundary "or" match (only inside "order")
    assert!((score.atomicity - 1.0).abs() < f64::EPSILON);
}

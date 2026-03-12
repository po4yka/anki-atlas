//! Skill relevance classification for flashcards.
//!
//! Classifies cards and topics as "alive" (high-value reasoning skills),
//! "dead" (rote memorization), or "neutral" based on topic tags and
//! content heuristics.

use serde::{Deserialize, Serialize};

/// Skill relevance classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkillRelevance {
    /// High-value: system design, debugging, reasoning, product thinking.
    Alive,
    /// Default: no strong signal either way.
    Neutral,
    /// Low-value: syntax recall, boilerplate memorization, rote lookup.
    Dead,
}

impl std::fmt::Display for SkillRelevance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Alive => write!(f, "alive"),
            Self::Neutral => write!(f, "neutral"),
            Self::Dead => write!(f, "dead"),
        }
    }
}

/// Topic-level relevance based on canonical tag.
static TOPIC_RELEVANCE: phf::Map<&'static str, SkillRelevance> = phf::phf_map! {
    // Dead: syntax recall, rote memorization
    "kotlin_syntax" => SkillRelevance::Dead,
    "kotlin_equality" => SkillRelevance::Dead,

    // Alive: system design and architecture
    "cs_system_design" => SkillRelevance::Alive,
    "cs_architecture" => SkillRelevance::Alive,
    "cs_distributed_systems" => SkillRelevance::Alive,
    "cs_patterns" => SkillRelevance::Alive,
    "android_architecture" => SkillRelevance::Alive,
    "android_debugging" => SkillRelevance::Alive,
    "cs_testing" => SkillRelevance::Alive,
    "android_testing" => SkillRelevance::Alive,
    "cs_concurrency" => SkillRelevance::Alive,
    "cs_security" => SkillRelevance::Alive,
    "kotlin_patterns" => SkillRelevance::Alive,
    "kotlin_concurrency" => SkillRelevance::Alive,
    "kotlin_coroutines" => SkillRelevance::Alive,
    "kotlin_error_handling" => SkillRelevance::Alive,
    "kotlin_performance" => SkillRelevance::Alive,
};

/// Look up the default relevance for a canonical topic tag.
///
/// Returns `Neutral` for unknown tags.
pub fn topic_relevance(canonical_tag: &str) -> SkillRelevance {
    TOPIC_RELEVANCE
        .get(canonical_tag)
        .copied()
        .unwrap_or(SkillRelevance::Neutral)
}

/// Dead-signal patterns in card front (question side).
const DEAD_FRONT_PATTERNS: &[&str] = &[
    "what is the syntax",
    "how do you write a",
    "what does the following code output",
    "write the code for",
    "what is the correct way to write",
    "fill in the blank",
    "complete the code",
    "what keyword",
    "what operator",
    "what symbol",
];

/// Alive-signal patterns in card front (question side).
const ALIVE_FRONT_PATTERNS: &[&str] = &[
    "why would you",
    "when would you",
    "what tradeoffs",
    "what trade-offs",
    "how would you design",
    "how would you debug",
    "compare",
    "what goes wrong if",
    "what happens when",
    "what problem does",
    "what are the implications",
    "how would you architect",
    "when should you prefer",
    "what is the impact of",
    "debug this",
];

/// Detect skill relevance from card content using heuristics.
///
/// Examines the question (front) for dead/alive signal patterns and
/// checks whether the answer (back) is pure code without explanation.
pub fn content_relevance(front: &str, back: &str) -> SkillRelevance {
    let front_lower = front.to_lowercase();
    let back_lower = back.to_lowercase();

    let dead_score = dead_content_score(&front_lower, &back_lower);
    let alive_score = alive_content_score(&front_lower);

    if alive_score > dead_score {
        SkillRelevance::Alive
    } else if dead_score > alive_score && dead_score >= 2 {
        SkillRelevance::Dead
    } else {
        SkillRelevance::Neutral
    }
}

fn dead_content_score(front: &str, back: &str) -> u32 {
    let mut score = 0u32;

    // Front matches dead patterns
    for pattern in DEAD_FRONT_PATTERNS {
        if front.contains(pattern) {
            score += 3;
            break;
        }
    }

    // Back is mostly a code block with little prose
    if is_code_only_answer(back) {
        score += 2;
    }

    score
}

fn alive_content_score(front: &str) -> u32 {
    let mut score = 0u32;

    for pattern in ALIVE_FRONT_PATTERNS {
        if front.contains(pattern) {
            score += 3;
            break;
        }
    }

    score
}

/// Check if the answer is predominantly code with minimal prose.
///
/// Returns true if the back content is mostly inside code fences
/// with fewer than 10 words of prose outside them.
fn is_code_only_answer(back: &str) -> bool {
    if back.is_empty() {
        return false;
    }

    let mut in_code_block = false;
    let mut prose_words = 0usize;
    let mut has_code = false;

    for line in back.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```")
            || trimmed.starts_with("<pre>")
            || trimmed.starts_with("<code>")
        {
            in_code_block = !in_code_block;
            if in_code_block {
                has_code = true;
            }
            continue;
        }
        if trimmed.starts_with("</pre>") || trimmed.starts_with("</code>") {
            in_code_block = false;
            continue;
        }
        if !in_code_block && !trimmed.is_empty() {
            prose_words += trimmed.split_whitespace().count();
        }
    }

    has_code && prose_words < 10
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- topic_relevance ---

    #[test]
    fn dead_topic() {
        assert_eq!(topic_relevance("kotlin_syntax"), SkillRelevance::Dead);
        assert_eq!(topic_relevance("kotlin_equality"), SkillRelevance::Dead);
    }

    #[test]
    fn alive_topic() {
        assert_eq!(topic_relevance("cs_system_design"), SkillRelevance::Alive);
        assert_eq!(topic_relevance("cs_architecture"), SkillRelevance::Alive);
        assert_eq!(
            topic_relevance("cs_distributed_systems"),
            SkillRelevance::Alive
        );
    }

    #[test]
    fn neutral_topic_unknown() {
        assert_eq!(
            topic_relevance("kotlin_collections"),
            SkillRelevance::Neutral
        );
        assert_eq!(topic_relevance("unknown_tag"), SkillRelevance::Neutral);
    }

    // --- content_relevance ---

    #[test]
    fn dead_content_syntax_recall() {
        let front = "What is the syntax for a when expression in Kotlin?";
        let back =
            "```kotlin\nwhen (x) {\n  1 -> print(\"one\")\n  else -> print(\"other\")\n}\n```";
        assert_eq!(content_relevance(front, back), SkillRelevance::Dead);
    }

    #[test]
    fn alive_content_design_question() {
        let front = "How would you design a caching layer for a mobile app?";
        let back = "Use LRU cache with TTL. Consider memory vs disk trade-offs.";
        assert_eq!(content_relevance(front, back), SkillRelevance::Alive);
    }

    #[test]
    fn alive_content_tradeoff_question() {
        let front = "What tradeoffs exist between coroutines and RxJava?";
        let back = "Coroutines are simpler, RxJava has richer operators.";
        assert_eq!(content_relevance(front, back), SkillRelevance::Alive);
    }

    #[test]
    fn alive_content_debug_question() {
        let front = "How would you debug a memory leak in an Android app?";
        let back = "Use LeakCanary and heap dumps to find retained references.";
        assert_eq!(content_relevance(front, back), SkillRelevance::Alive);
    }

    #[test]
    fn neutral_content_no_signals() {
        let front = "What is a coroutine scope?";
        let back = "A coroutine scope defines the lifecycle of coroutines launched within it.";
        assert_eq!(content_relevance(front, back), SkillRelevance::Neutral);
    }

    #[test]
    fn dead_content_code_only_answer() {
        let front = "How do you write a singleton in Kotlin?";
        let back = "```kotlin\nobject MySingleton {\n  val x = 1\n}\n```";
        assert_eq!(content_relevance(front, back), SkillRelevance::Dead);
    }

    #[test]
    fn neutral_content_code_with_explanation() {
        let front = "What is a Kotlin object declaration?";
        let back = "Use the object declaration. This ensures thread-safe lazy initialization \
                     and a single instance across the entire application. The JVM guarantees \
                     that the object is created only once.\n\
                     ```kotlin\nobject MySingleton {\n  val x = 1\n}\n```";
        // No dead front pattern + enough prose = neutral
        assert_eq!(content_relevance(front, back), SkillRelevance::Neutral);
    }

    // --- is_code_only_answer ---

    #[test]
    fn empty_answer_not_code_only() {
        assert!(!is_code_only_answer(""));
    }

    #[test]
    fn pure_code_block_is_code_only() {
        let back = "```\nfn main() {}\n```";
        assert!(is_code_only_answer(back));
    }

    #[test]
    fn code_with_prose_is_not_code_only() {
        let back = "This function demonstrates the builder pattern which provides a fluent \
                     API for constructing complex objects step by step.\n```\nfn main() {}\n```";
        assert!(!is_code_only_answer(back));
    }
}

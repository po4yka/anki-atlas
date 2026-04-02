use generator::{
    GeneratedCard, GenerationDeps, GenerationResult, Severity, SplitDecision, SplitPlan,
    ValidationIssue, ValidationResult,
};
use sha2::{Digest, Sha256};

// --- GeneratedCard ---

#[test]
fn generated_card_content_hash_is_sha256_first_16_hex() {
    let apf_html = "<div>test card</div>";
    let expected_hash = {
        let mut hasher = Sha256::new();
        hasher.update(apf_html.as_bytes());
        let result = hasher.finalize();
        format!("{:x}", result)[..16].to_string()
    };

    let card = GeneratedCard {
        card_index: 1,
        slug: "test-card-1-en".to_string(),
        lang: "en".to_string(),
        apf_html: apf_html.to_string(),
        confidence: 0.95,
        content_hash: expected_hash.clone(),
    };

    assert_eq!(card.content_hash, expected_hash);
    assert_eq!(card.content_hash.len(), 16);
}

#[test]
fn generated_card_serialization_roundtrip() {
    let card = GeneratedCard {
        card_index: 0,
        slug: "rust-ownership-1-en".to_string(),
        lang: "en".to_string(),
        apf_html: "<p>What is ownership?</p>".to_string(),
        confidence: 0.85,
        content_hash: "abcdef0123456789".to_string(),
    };

    let json = serde_json::to_string(&card).expect("serialize");
    let deserialized: GeneratedCard = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.slug, card.slug);
    assert_eq!(deserialized.card_index, card.card_index);
    assert_eq!(deserialized.confidence, card.confidence);
    assert_eq!(deserialized.content_hash, card.content_hash);
}

#[test]
fn generated_card_confidence_range() {
    let card = GeneratedCard {
        card_index: 0,
        slug: "test".to_string(),
        lang: "en".to_string(),
        apf_html: "html".to_string(),
        confidence: 0.0,
        content_hash: String::new(),
    };
    assert!(card.confidence >= 0.0 && card.confidence <= 1.0);
}

// --- GenerationResult ---

#[test]
fn generation_result_tracks_card_count_and_time() {
    let result = GenerationResult {
        cards: vec![],
        total_cards: 0,
        model_used: "gpt-4".to_string(),
        generation_time_secs: 2.5,
        warnings: vec!["low confidence".to_string()],
    };

    assert_eq!(result.total_cards, 0);
    assert_eq!(result.model_used, "gpt-4");
    assert!((result.generation_time_secs - 2.5).abs() < f64::EPSILON);
    assert_eq!(result.warnings.len(), 1);
}

#[test]
fn generation_result_serialization_roundtrip() {
    let result = GenerationResult {
        cards: vec![],
        total_cards: 3,
        model_used: "claude-3".to_string(),
        generation_time_secs: 1.2,
        warnings: vec![],
    };

    let json = serde_json::to_string(&result).expect("serialize");
    let deserialized: GenerationResult = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.total_cards, 3);
    assert_eq!(deserialized.model_used, "claude-3");
}

// --- GenerationDeps ---

#[test]
fn generation_deps_construction() {
    let deps = GenerationDeps {
        note_title: "Rust Ownership".to_string(),
        topic: "ownership".to_string(),
        language_tags: vec!["en".to_string(), "ru".to_string()],
        source_file: "notes/rust.md".to_string(),
        skill_bias: None,
    };

    assert_eq!(deps.note_title, "Rust Ownership");
    assert_eq!(deps.language_tags.len(), 2);
}

#[test]
fn generation_deps_serialization_roundtrip() {
    let deps = GenerationDeps {
        note_title: "Test".to_string(),
        topic: "testing".to_string(),
        language_tags: vec!["en".to_string()],
        source_file: "test.md".to_string(),
        skill_bias: None,
    };

    let json = serde_json::to_string(&deps).expect("serialize");
    let deserialized: GenerationDeps = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.note_title, "Test");
}

// --- SplitPlan ---

#[test]
fn split_plan_construction() {
    let plan = SplitPlan {
        card_number: 1,
        concept: "ownership basics".to_string(),
        question: "What is ownership?".to_string(),
        answer_summary: "A memory management concept".to_string(),
    };

    assert_eq!(plan.card_number, 1);
    assert_eq!(plan.concept, "ownership basics");
}

// --- SplitDecision ---

#[test]
fn split_decision_no_split() {
    let decision = SplitDecision {
        should_split: false,
        card_count: 1,
        plans: vec![],
        reasoning: "Content is atomic".to_string(),
    };

    assert!(!decision.should_split);
    assert_eq!(decision.card_count, 1);
    assert!(decision.plans.is_empty());
}

#[test]
fn split_decision_with_split() {
    let decision = SplitDecision {
        should_split: true,
        card_count: 3,
        plans: vec![
            SplitPlan {
                card_number: 1,
                concept: "concept A".to_string(),
                question: "Q1".to_string(),
                answer_summary: "A1".to_string(),
            },
            SplitPlan {
                card_number: 2,
                concept: "concept B".to_string(),
                question: "Q2".to_string(),
                answer_summary: "A2".to_string(),
            },
            SplitPlan {
                card_number: 3,
                concept: "concept C".to_string(),
                question: "Q3".to_string(),
                answer_summary: "A3".to_string(),
            },
        ],
        reasoning: "Content covers multiple distinct concepts".to_string(),
    };

    assert!(decision.should_split);
    assert_eq!(decision.card_count, 3);
    assert_eq!(decision.plans.len(), 3);
}

#[test]
fn split_decision_serialization_roundtrip() {
    let decision = SplitDecision {
        should_split: true,
        card_count: 2,
        plans: vec![SplitPlan {
            card_number: 1,
            concept: "test".to_string(),
            question: "Q".to_string(),
            answer_summary: "A".to_string(),
        }],
        reasoning: "reason".to_string(),
    };

    let json = serde_json::to_string(&decision).expect("serialize");
    let deserialized: SplitDecision = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.should_split, true);
    assert_eq!(deserialized.plans.len(), 1);
}

// --- Severity ---

#[test]
fn severity_equality() {
    assert_eq!(Severity::Error, Severity::Error);
    assert_eq!(Severity::Warning, Severity::Warning);
    assert_ne!(Severity::Error, Severity::Warning);
}

#[test]
fn severity_serialization() {
    let json = serde_json::to_string(&Severity::Error).expect("serialize");
    let deserialized: Severity = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized, Severity::Error);
}

// --- ValidationIssue ---

#[test]
fn validation_issue_with_location() {
    let issue = ValidationIssue {
        severity: Severity::Error,
        message: "missing title".to_string(),
        location: Some("line 5".to_string()),
    };

    assert_eq!(issue.severity, Severity::Error);
    assert_eq!(issue.location, Some("line 5".to_string()));
}

#[test]
fn validation_issue_without_location() {
    let issue = ValidationIssue {
        severity: Severity::Warning,
        message: "low confidence".to_string(),
        location: None,
    };

    assert_eq!(issue.severity, Severity::Warning);
    assert!(issue.location.is_none());
}

// --- ValidationResult ---

#[test]
fn validation_result_is_valid_with_no_issues() {
    let result = ValidationResult { issues: vec![] };
    assert!(result.is_valid());
}

#[test]
fn validation_result_is_valid_with_only_warnings() {
    let result = ValidationResult {
        issues: vec![ValidationIssue {
            severity: Severity::Warning,
            message: "minor issue".to_string(),
            location: None,
        }],
    };
    assert!(result.is_valid());
}

#[test]
fn validation_result_is_invalid_with_error() {
    let result = ValidationResult {
        issues: vec![ValidationIssue {
            severity: Severity::Error,
            message: "critical issue".to_string(),
            location: None,
        }],
    };
    assert!(!result.is_valid());
}

#[test]
fn validation_result_is_invalid_with_mixed_issues() {
    let result = ValidationResult {
        issues: vec![
            ValidationIssue {
                severity: Severity::Warning,
                message: "minor".to_string(),
                location: None,
            },
            ValidationIssue {
                severity: Severity::Error,
                message: "critical".to_string(),
                location: None,
            },
        ],
    };
    assert!(!result.is_valid());
}

#[test]
fn validation_result_errors_filters_correctly() {
    let result = ValidationResult {
        issues: vec![
            ValidationIssue {
                severity: Severity::Warning,
                message: "warn1".to_string(),
                location: None,
            },
            ValidationIssue {
                severity: Severity::Error,
                message: "err1".to_string(),
                location: None,
            },
            ValidationIssue {
                severity: Severity::Error,
                message: "err2".to_string(),
                location: None,
            },
        ],
    };

    let errors = result.errors();
    assert_eq!(errors.len(), 2);
    assert!(errors.iter().all(|e| e.severity == Severity::Error));
}

#[test]
fn validation_result_warnings_filters_correctly() {
    let result = ValidationResult {
        issues: vec![
            ValidationIssue {
                severity: Severity::Warning,
                message: "warn1".to_string(),
                location: None,
            },
            ValidationIssue {
                severity: Severity::Error,
                message: "err1".to_string(),
                location: None,
            },
            ValidationIssue {
                severity: Severity::Warning,
                message: "warn2".to_string(),
                location: None,
            },
        ],
    };

    let warnings = result.warnings();
    assert_eq!(warnings.len(), 2);
    assert!(warnings.iter().all(|w| w.severity == Severity::Warning));
}

#[test]
fn validation_result_serialization_roundtrip() {
    let result = ValidationResult {
        issues: vec![ValidationIssue {
            severity: Severity::Error,
            message: "test".to_string(),
            location: Some("loc".to_string()),
        }],
    };

    let json = serde_json::to_string(&result).expect("serialize");
    let deserialized: ValidationResult = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.issues.len(), 1);
    assert!(!deserialized.is_valid());
}

// --- Send + Sync assertions ---

common::assert_send_sync!(
    GeneratedCard,
    GenerationResult,
    GenerationDeps,
    SplitPlan,
    SplitDecision,
    Severity,
    ValidationIssue,
    ValidationResult,
);

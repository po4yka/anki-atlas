use anki_atlas_mcp::tools::*;

// --- Constants ---

#[test]
fn tool_timeout_is_30_seconds() {
    assert_eq!(TOOL_TIMEOUT_SECS, 30);
}

#[test]
fn sync_timeout_is_120_seconds() {
    assert_eq!(SYNC_TIMEOUT_SECS, 120);
}

#[test]
fn index_timeout_is_300_seconds() {
    assert_eq!(INDEX_TIMEOUT_SECS, 300);
}

// --- SearchInput ---

#[test]
fn search_input_deserialize_minimal() {
    let json = r#"{"query": "hello"}"#;
    let input: SearchInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.query, "hello");
    assert_eq!(input.limit, 20); // default
    assert!(input.deck_filter.is_none());
    assert!(input.tag_filter.is_none());
    assert!(!input.semantic_only);
    assert!(!input.fts_only);
}

#[test]
fn search_input_deserialize_full() {
    let json = r#"{
        "query": "calculus",
        "limit": 5,
        "deck_filter": ["Math"],
        "tag_filter": ["math::calculus"],
        "semantic_only": true,
        "fts_only": false
    }"#;
    let input: SearchInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.query, "calculus");
    assert_eq!(input.limit, 5);
    assert_eq!(input.deck_filter.as_ref().unwrap(), &["Math"]);
    assert_eq!(input.tag_filter.as_ref().unwrap(), &["math::calculus"]);
    assert!(input.semantic_only);
    assert!(!input.fts_only);
}

// --- TopicCoverageInput ---

#[test]
fn topic_coverage_input_defaults() {
    let json = r#"{"topic_path": "math/calculus"}"#;
    let input: TopicCoverageInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.topic_path, "math/calculus");
    assert!(input.include_subtree); // default true
}

#[test]
fn topic_coverage_input_explicit_false() {
    let json = r#"{"topic_path": "cs", "include_subtree": false}"#;
    let input: TopicCoverageInput = serde_json::from_str(json).unwrap();
    assert!(!input.include_subtree);
}

// --- TopicGapsInput ---

#[test]
fn topic_gaps_input_defaults() {
    let json = r#"{"topic_path": "math"}"#;
    let input: TopicGapsInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.topic_path, "math");
    assert_eq!(input.min_coverage, 1); // default
}

#[test]
fn topic_gaps_input_custom_min_coverage() {
    let json = r#"{"topic_path": "cs", "min_coverage": 5}"#;
    let input: TopicGapsInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.min_coverage, 5);
}

// --- DuplicatesInput ---

#[test]
fn duplicates_input_defaults() {
    let json = r#"{}"#;
    let input: DuplicatesInput = serde_json::from_str(json).unwrap();
    assert!((input.threshold - 0.92).abs() < f64::EPSILON); // default
    assert_eq!(input.max_clusters, 50); // default
    assert!(input.deck_filter.is_none());
    assert!(input.tag_filter.is_none());
}

#[test]
fn duplicates_input_custom() {
    let json = r#"{
        "threshold": 0.85,
        "max_clusters": 100,
        "deck_filter": ["Default"],
        "tag_filter": ["cs"]
    }"#;
    let input: DuplicatesInput = serde_json::from_str(json).unwrap();
    assert!((input.threshold - 0.85).abs() < f64::EPSILON);
    assert_eq!(input.max_clusters, 100);
    assert!(input.deck_filter.is_some());
    assert!(input.tag_filter.is_some());
}

// --- SyncInput ---

#[test]
fn sync_input_defaults() {
    let json = r#"{"collection_path": "/path/to/collection.anki2"}"#;
    let input: SyncInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.collection_path, "/path/to/collection.anki2");
    assert!(!input.run_index); // default false
}

#[test]
fn sync_input_with_index() {
    let json = r#"{"collection_path": "/path/to/collection.anki2", "run_index": true}"#;
    let input: SyncInput = serde_json::from_str(json).unwrap();
    assert!(input.run_index);
}

// --- GenerateInput ---

#[test]
fn generate_input_minimal() {
    let json = r##"{"text": "# My Note"}"##;
    let input: GenerateInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.text, "# My Note");
    assert!(input.deck.is_none());
}

#[test]
fn generate_input_with_deck() {
    let json = r#"{"text": "content", "deck": "CS::Algorithms"}"#;
    let input: GenerateInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.deck.as_deref(), Some("CS::Algorithms"));
}

// --- ValidateInput ---

#[test]
fn validate_input_defaults() {
    let json = r#"{"front": "What is X?", "back": "Y"}"#;
    let input: ValidateInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.front, "What is X?");
    assert_eq!(input.back, "Y");
    assert!(input.tags.is_none());
    assert!(input.check_quality); // default true
}

#[test]
fn validate_input_full() {
    let json = r#"{
        "front": "Q?",
        "back": "A",
        "tags": ["cs::algo"],
        "check_quality": false
    }"#;
    let input: ValidateInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.tags.as_ref().unwrap(), &["cs::algo"]);
    assert!(!input.check_quality);
}

// --- ObsidianSyncInput ---

#[test]
fn obsidian_sync_input_defaults() {
    let json = r#"{"vault_path": "/home/user/vault"}"#;
    let input: ObsidianSyncInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.vault_path, "/home/user/vault");
    assert!(input.dry_run); // default true
}

#[test]
fn obsidian_sync_input_wet_run() {
    let json = r#"{"vault_path": "/vault", "dry_run": false}"#;
    let input: ObsidianSyncInput = serde_json::from_str(json).unwrap();
    assert!(!input.dry_run);
}

// --- TagAuditInput ---

#[test]
fn tag_audit_input_defaults() {
    let json = r#"{"tags": ["cs::algo", "math"]}"#;
    let input: TagAuditInput = serde_json::from_str(json).unwrap();
    assert_eq!(input.tags, vec!["cs::algo", "math"]);
    assert!(!input.fix); // default false
}

#[test]
fn tag_audit_input_with_fix() {
    let json = r#"{"tags": ["Bad_Tag"], "fix": true}"#;
    let input: TagAuditInput = serde_json::from_str(json).unwrap();
    assert!(input.fix);
}

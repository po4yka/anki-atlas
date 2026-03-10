use anki_atlas_mcp::tools::*;

// --- Constants ---

#[test]
fn tool_timeout_is_30_seconds() {
    assert_eq!(TOOL_TIMEOUT_SECS, 30);
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

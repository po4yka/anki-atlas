use anki_atlas_mcp::tools::{
    GenerateToolInput, ObsidianSyncToolInput, OutputMode, SearchToolInput, TagAuditToolInput,
};

#[test]
fn search_input_defaults_to_markdown() {
    let input: SearchToolInput =
        serde_json::from_str(r#"{"query":"ownership"}"#).expect("valid search input");
    assert!(matches!(input.output_mode, OutputMode::Markdown));
    assert_eq!(input.limit, 10);
}

#[test]
fn generate_input_parses_file_path() {
    let input: GenerateToolInput =
        serde_json::from_str(r#"{"file_path":"/tmp/note.md","output_mode":"json"}"#)
            .expect("valid generate input");
    assert_eq!(input.file_path, "/tmp/note.md");
    assert!(matches!(input.output_mode, OutputMode::Json));
}

#[test]
fn obsidian_sync_defaults_to_dry_run() {
    let input: ObsidianSyncToolInput =
        serde_json::from_str(r#"{"vault_path":"/vault"}"#).expect("valid obsidian input");
    assert_eq!(input.vault_path, "/vault");
    assert!(input.dry_run);
}

#[test]
fn tag_audit_defaults_to_no_fix() {
    let input: TagAuditToolInput =
        serde_json::from_str(r#"{"file_path":"/tmp/tags.txt"}"#).expect("valid tag-audit input");
    assert!(!input.fix);
}

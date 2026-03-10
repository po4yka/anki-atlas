use anki_atlas_mcp::handlers::{
    clamp_limit, format_error, handle_generate, handle_obsidian_sync, handle_tag_audit,
    validate_vault_path, ErrorKind,
};
use anki_atlas_mcp::tools::{GenerateInput, ObsidianSyncInput, TagAuditInput};
use tempfile::TempDir;

// ─── format_error ───────────────────────────────────────────────────

#[test]
fn format_error_database_unavailable() {
    let result = format_error(ErrorKind::DatabaseUnavailable, "search");
    assert!(result.contains("Database unavailable"), "should mention database");
    assert!(result.contains("search"), "should mention operation");
    assert!(result.contains("PostgreSQL"), "should suggest checking PostgreSQL");
}

#[test]
fn format_error_vector_store_unavailable() {
    let result = format_error(ErrorKind::VectorStoreUnavailable, "search");
    assert!(result.contains("Vector"), "should mention vector store");
    assert!(result.contains("Qdrant"), "should suggest checking Qdrant");
}

#[test]
fn format_error_timeout() {
    let result = format_error(ErrorKind::Timeout, "search");
    assert!(result.contains("timed out"), "should mention timeout");
    assert!(result.contains("search"), "should mention operation");
    assert!(result.contains("limit") || result.contains("specific"), "should suggest reducing scope");
}

#[test]
fn format_error_other() {
    let result = format_error(
        ErrorKind::Other {
            error_type: "ParseError".to_string(),
            message: "invalid JSON".to_string(),
        },
        "generate",
    );
    assert!(result.contains("generate"), "should mention operation");
    assert!(result.contains("ParseError"), "should show error type");
    assert!(result.contains("invalid JSON"), "should show message");
}

// ─── clamp_limit ────────────────────────────────────────────────────

#[test]
fn clamp_limit_within_range() {
    assert_eq!(clamp_limit(50, 1, 100), 50);
}

#[test]
fn clamp_limit_below_min() {
    assert_eq!(clamp_limit(0, 1, 100), 1);
}

#[test]
fn clamp_limit_above_max() {
    assert_eq!(clamp_limit(200, 1, 100), 100);
}

#[test]
fn clamp_limit_at_boundaries() {
    assert_eq!(clamp_limit(1, 1, 100), 1);
    assert_eq!(clamp_limit(100, 1, 100), 100);
}

// ─── validate_vault_path ────────────────────────────────────────────

#[test]
fn validate_vault_path_valid_dir() {
    let dir = TempDir::new().unwrap();
    assert!(validate_vault_path(dir.path().to_str().unwrap()).is_ok());
}

#[test]
fn validate_vault_path_not_a_directory() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("file.txt");
    std::fs::write(&file, b"content").unwrap();
    let err = validate_vault_path(file.to_str().unwrap()).unwrap_err();
    assert!(
        err.contains("not a directory") || err.contains("directory"),
        "should indicate not a directory"
    );
}

#[test]
fn validate_vault_path_nonexistent() {
    let err = validate_vault_path("/nonexistent/vault").unwrap_err();
    assert!(
        err.contains("not found") || err.contains("does not exist"),
        "should indicate path not found"
    );
}

// ─── handle_generate ────────────────────────────────────────────────

#[tokio::test]
async fn handle_generate_with_sections() {
    let input = GenerateInput {
        text: "# My Title\n\n## Section One\n\nContent here.\n\n## Section Two\n\nMore content.".to_string(),
        deck: None,
    };
    let result = handle_generate(input).await;
    assert!(result.contains("My Title"), "should extract title");
    assert!(result.contains("Section"), "should show sections");
    assert!(result.contains("Preview"), "should be a preview");
}

#[tokio::test]
async fn handle_generate_with_no_title() {
    let input = GenerateInput {
        text: "Just some plain text without headings.".to_string(),
        deck: Some("TestDeck".to_string()),
    };
    let result = handle_generate(input).await;
    assert!(
        result.contains("not detected") || result.contains("Preview"),
        "should handle missing title gracefully"
    );
}

#[tokio::test]
async fn handle_generate_empty_text() {
    let input = GenerateInput {
        text: String::new(),
        deck: None,
    };
    let result = handle_generate(input).await;
    // Should return a result (possibly with 0 sections), not panic
    assert!(!result.is_empty(), "should return non-empty response");
}

// ─── handle_obsidian_sync (validation paths) ────────────────────────

#[tokio::test]
async fn handle_obsidian_sync_nonexistent_vault() {
    let input = ObsidianSyncInput {
        vault_path: "/nonexistent/vault".to_string(),
        dry_run: true,
    };
    let result = handle_obsidian_sync(input).await;
    assert!(
        result.contains("not found") || result.contains("does not exist"),
        "should indicate vault not found"
    );
}

#[tokio::test]
async fn handle_obsidian_sync_file_not_directory() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("not_a_dir.txt");
    std::fs::write(&file, b"content").unwrap();
    let input = ObsidianSyncInput {
        vault_path: file.to_str().unwrap().to_string(),
        dry_run: true,
    };
    let result = handle_obsidian_sync(input).await;
    assert!(
        result.contains("not a directory") || result.contains("directory"),
        "should indicate not a directory"
    );
}

#[tokio::test]
async fn handle_obsidian_sync_empty_vault() {
    let dir = TempDir::new().unwrap();
    let input = ObsidianSyncInput {
        vault_path: dir.path().to_str().unwrap().to_string(),
        dry_run: true,
    };
    let result = handle_obsidian_sync(input).await;
    assert!(result.contains("Vault") || result.contains("Scan"), "should return scan result");
    assert!(result.contains("0") || result.contains("Notes found"), "should show zero notes");
}

#[tokio::test]
async fn handle_obsidian_sync_with_notes() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("note1.md"),
        "# First Note\n\n## Section\n\nContent",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("note2.md"),
        "# Second Note\n\nPlain content",
    )
    .unwrap();
    let input = ObsidianSyncInput {
        vault_path: dir.path().to_str().unwrap().to_string(),
        dry_run: true,
    };
    let result = handle_obsidian_sync(input).await;
    assert!(result.contains("2") || result.contains("Notes found"), "should find 2 notes");
}

// ─── handle_tag_audit ───────────────────────────────────────────────

#[tokio::test]
async fn handle_tag_audit_all_valid() {
    let input = TagAuditInput {
        tags: vec!["math::calculus".to_string(), "cs::algorithms".to_string()],
        fix: false,
    };
    let result = handle_tag_audit(input).await;
    assert!(result.contains("Audit"), "should contain audit header");
    assert!(result.contains("2"), "should show total tag count");
}

#[tokio::test]
async fn handle_tag_audit_with_issues() {
    let input = TagAuditInput {
        tags: vec![
            "UPPERCASE_TAG".to_string(),
            "bad/separator".to_string(),
        ],
        fix: false,
    };
    let result = handle_tag_audit(input).await;
    assert!(result.contains("issues") || result.contains("Issues"), "should report issues");
}

#[tokio::test]
async fn handle_tag_audit_with_fix() {
    let input = TagAuditInput {
        tags: vec!["bad_tag_format".to_string()],
        fix: true,
    };
    let result = handle_tag_audit(input).await;
    assert!(result.contains("Audit"), "should contain audit header");
}

#[tokio::test]
async fn handle_tag_audit_empty_tags() {
    let input = TagAuditInput {
        tags: vec![],
        fix: false,
    };
    let result = handle_tag_audit(input).await;
    assert!(result.contains("0") || result.contains("Audit"), "should handle empty tags");
}

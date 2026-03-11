use std::fs;

use surface_runtime::ObsidianScanService;

#[test]
fn scan_non_dry_run_returns_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    let service = ObsidianScanService::new();
    let result = service.scan(dir.path(), &[], false);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("unsupported"), "got: {err}");
}

#[test]
fn scan_nonexistent_vault_returns_path_not_found() {
    let service = ObsidianScanService::new();
    let result = service.scan(
        std::path::Path::new("/tmp/nonexistent_vault_xyz"),
        &[],
        true,
    );
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("path not found"), "got: {err}");
}

#[test]
fn scan_empty_vault_returns_zero_notes() {
    let dir = tempfile::tempdir().unwrap();
    let service = ObsidianScanService::new();
    let preview = service.scan(dir.path(), &[], true).unwrap();
    assert_eq!(preview.note_count, 0);
    assert_eq!(preview.generated_cards, 0);
    assert!(preview.notes.is_empty());
}

#[test]
fn scan_vault_with_markdown_notes() {
    let dir = tempfile::tempdir().unwrap();
    let notes_dir = dir.path().join("notes");
    fs::create_dir_all(&notes_dir).unwrap();
    fs::write(
        notes_dir.join("topic.md"),
        "---\ntitle: Topic\n---\n## Section 1\nContent here.\n",
    )
    .unwrap();
    fs::write(
        notes_dir.join("another.md"),
        "---\ntitle: Another\n---\n## Part A\nMore content.\n",
    )
    .unwrap();

    let service = ObsidianScanService::new();
    let preview = service
        .scan(dir.path(), &["notes".to_string()], true)
        .unwrap();
    assert!(preview.note_count >= 2, "should find at least 2 notes");
    assert!(!preview.notes.is_empty());
}

use std::fs;

use surface_runtime::ValidationService;

#[test]
fn validate_nonexistent_file_returns_path_not_found() {
    let service = ValidationService::new();
    let result = service.validate_file(std::path::Path::new("/tmp/nonexistent_val.txt"), false);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("path not found"), "got: {err}");
}

#[test]
fn validate_valid_content_passes() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("card.txt");
    fs::write(
        &file,
        "What is Rust?\n---\nRust is a systems programming language focused on safety.\n---\ncs::rust\n",
    )
    .unwrap();

    let service = ValidationService::new();
    let summary = service.validate_file(&file, false).unwrap();
    assert_eq!(summary.source_file, file);
    assert!(summary.is_valid, "valid content should pass validation");
    assert!(summary.quality.is_none());
}

#[test]
fn validate_missing_back_returns_invalid_input() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("bad.txt");
    fs::write(&file, "Only front content, no separator").unwrap();

    let service = ValidationService::new();
    let result = service.validate_file(&file, false);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("invalid input"), "got: {err}");
}

#[test]
fn validate_include_quality_returns_score() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("quality.txt");
    fs::write(
        &file,
        "What is ownership in Rust?\n---\nOwnership is Rust's system for managing memory without a garbage collector.\n---\ncs::rust\n",
    )
    .unwrap();

    let service = ValidationService::new();
    let summary = service.validate_file(&file, true).unwrap();
    assert!(summary.quality.is_some(), "quality should be computed");
}

#[test]
fn validate_without_quality_returns_none() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("no_quality.txt");
    fs::write(&file, "Front\n---\nBack\n---\ncs::test\n").unwrap();

    let service = ValidationService::new();
    let summary = service.validate_file(&file, false).unwrap();
    assert!(summary.quality.is_none());
}

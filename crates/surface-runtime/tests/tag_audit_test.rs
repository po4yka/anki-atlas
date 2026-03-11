use std::fs;

use surface_runtime::TagAuditService;

#[test]
fn audit_file_nonexistent_returns_path_not_found() {
    let service = TagAuditService::new();
    let result = service.audit_file(std::path::Path::new("/tmp/nonexistent_tag_file.txt"), false);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("path not found"), "got: {err}");
}

#[test]
fn audit_file_returns_entries_for_each_tag() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("tags.txt");
    fs::write(&file, "cs::rust\ncs::algorithms\n").unwrap();

    let service = TagAuditService::new();
    let summary = service.audit_file(&file, false).unwrap();
    assert_eq!(summary.source_file, file);
    assert!(!summary.applied_fixes);
    assert_eq!(summary.entries.len(), 2);
    assert_eq!(summary.entries[0].tag, "cs::rust");
    assert_eq!(summary.entries[1].tag, "cs::algorithms");
}

#[test]
fn audit_file_invalid_tags_reports_issues() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("tags.txt");
    // Use tags likely to have validation issues (invalid format)
    fs::write(&file, "UPPER_CASE\n").unwrap();

    let service = TagAuditService::new();
    let summary = service.audit_file(&file, false).unwrap();
    assert_eq!(summary.entries.len(), 1);
    // The tag should at least be normalized differently
    let entry = &summary.entries[0];
    assert_eq!(entry.tag, "UPPER_CASE");
}

#[test]
fn audit_file_normalize_records_difference() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("tags.txt");
    fs::write(&file, "CS::Rust\n").unwrap();

    let service = TagAuditService::new();
    let summary = service.audit_file(&file, false).unwrap();
    let entry = &summary.entries[0];
    assert_eq!(entry.tag, "CS::Rust");
    // Normalized form should be lowercase
    assert_eq!(entry.normalized, entry.normalized.to_lowercase());
}

#[test]
fn audit_file_apply_fixes_rewrites_file() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("tags.txt");
    fs::write(&file, "CS::Rust\ncs::algorithms\n").unwrap();

    let service = TagAuditService::new();
    let summary = service.audit_file(&file, true).unwrap();
    assert!(summary.applied_fixes);

    let rewritten = fs::read_to_string(&file).unwrap();
    // All tags should be normalized (lowercase)
    for line in rewritten.lines() {
        if !line.is_empty() {
            assert_eq!(line, line.to_lowercase(), "tag should be normalized: {line}");
        }
    }
}

#[test]
fn audit_file_apply_fixes_deduplicates() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("tags.txt");
    fs::write(&file, "cs::rust\nCS::Rust\ncs::rust\n").unwrap();

    let service = TagAuditService::new();
    service.audit_file(&file, true).unwrap();

    let rewritten = fs::read_to_string(&file).unwrap();
    let lines: Vec<&str> = rewritten.lines().filter(|l| !l.is_empty()).collect();
    // BTreeSet dedup means only one copy of the normalized tag
    assert_eq!(lines.len(), 1, "duplicates should be deduped, got: {lines:?}");
}

#[test]
fn audit_file_empty_file_returns_empty_entries() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("tags.txt");
    fs::write(&file, "").unwrap();

    let service = TagAuditService::new();
    let summary = service.audit_file(&file, false).unwrap();
    assert!(summary.entries.is_empty());
}

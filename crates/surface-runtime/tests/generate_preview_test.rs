use std::fs;

use surface_runtime::GeneratePreviewService;

#[test]
fn preview_nonexistent_file_returns_path_not_found() {
    let service = GeneratePreviewService::new();
    let result = service.preview(std::path::Path::new("/tmp/nonexistent_note.md"));
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("path not found"), "got: {err}");
}

#[test]
fn preview_note_with_title_and_sections() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("note.md");
    fs::write(
        &file,
        "---\ntitle: Test Note\n---\n## Section 1\nContent one.\n## Section 2\nContent two.\n",
    )
    .unwrap();

    let service = GeneratePreviewService::new();
    let preview = service.preview(&file).unwrap();
    assert_eq!(preview.title.as_deref(), Some("Test Note"));
    assert!(preview.sections.len() >= 2);
    assert!(preview.warnings.is_empty());
}

#[test]
fn preview_note_without_title_generates_warning() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("untitled.md");
    fs::write(&file, "## Section\nSome content.\n").unwrap();

    let service = GeneratePreviewService::new();
    let preview = service.preview(&file).unwrap();
    assert!(preview.title.is_none());
    assert!(!preview.warnings.is_empty());
    assert!(preview.warnings[0].contains("No title"));
}

#[test]
fn preview_empty_sections_filtered() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("sparse.md");
    fs::write(
        &file,
        "---\ntitle: Sparse\n---\n## Filled\nReal content.\n## Empty\n\n",
    )
    .unwrap();

    let service = GeneratePreviewService::new();
    let preview = service.preview(&file).unwrap();
    // estimated_cards counts only non-empty sections
    assert!(preview.estimated_cards >= 1);
    // cards should only include non-empty content sections
    for card in &preview.cards {
        assert!(!card.apf_html.trim().is_empty());
    }
}

#[test]
fn preview_content_hash_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("determ.md");
    fs::write(&file, "---\ntitle: Deterministic\n---\n## S1\nContent.\n").unwrap();

    let service = GeneratePreviewService::new();
    let p1 = service.preview(&file).unwrap();
    let p2 = service.preview(&file).unwrap();
    assert!(!p1.cards.is_empty());
    assert_eq!(p1.cards[0].content_hash, p2.cards[0].content_hash);
}

#[test]
fn preview_card_slugs_are_sequential() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("multi.md");
    fs::write(
        &file,
        "---\ntitle: Multi\n---\n## A\nContent A.\n## B\nContent B.\n## C\nContent C.\n",
    )
    .unwrap();

    let service = GeneratePreviewService::new();
    let preview = service.preview(&file).unwrap();
    for (i, card) in preview.cards.iter().enumerate() {
        let expected_suffix = format!("-{}", i + 1);
        assert!(
            card.slug.ends_with(&expected_suffix),
            "slug {} should end with {}",
            card.slug,
            expected_suffix
        );
    }
}

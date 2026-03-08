use std::fs;
use std::os::unix::fs as unix_fs;
use std::path::PathBuf;

use tempfile::TempDir;

use super::*;

// ---------------------------------------------------------------------------
// parse_note: title extraction
// ---------------------------------------------------------------------------

#[test]
fn parse_note_title_from_frontmatter() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("note.md");
    fs::write(
        &path,
        "---\ntitle: My Title\n---\nSome body text.\n",
    )
    .unwrap();

    let note = parse_note(&path, None).unwrap();
    assert_eq!(note.title, Some("My Title".to_string()));
}

#[test]
fn parse_note_title_from_h1_heading() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("note.md");
    fs::write(&path, "# First Heading\nSome body text.\n").unwrap();

    let note = parse_note(&path, None).unwrap();
    assert_eq!(note.title, Some("First Heading".to_string()));
}

#[test]
fn parse_note_no_title() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("note.md");
    fs::write(&path, "Just some body text with no heading.\n").unwrap();

    let note = parse_note(&path, None).unwrap();
    assert_eq!(note.title, None);
}

#[test]
fn parse_note_frontmatter_title_takes_precedence_over_h1() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("note.md");
    fs::write(
        &path,
        "---\ntitle: FM Title\n---\n# Heading Title\nBody.\n",
    )
    .unwrap();

    let note = parse_note(&path, None).unwrap();
    assert_eq!(note.title, Some("FM Title".to_string()));
}

// ---------------------------------------------------------------------------
// parse_note: section splitting
// ---------------------------------------------------------------------------

#[test]
fn parse_note_splits_sections_by_heading() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("note.md");
    fs::write(
        &path,
        "# Section One\nContent one.\n## Section Two\nContent two.\n",
    )
    .unwrap();

    let note = parse_note(&path, None).unwrap();

    // Should have 2 sections (both headings start sections)
    assert!(note.sections.len() >= 2);
    assert_eq!(note.sections[0].0, "# Section One");
    assert!(note.sections[0].1.contains("Content one."));
    assert_eq!(note.sections[1].0, "## Section Two");
    assert!(note.sections[1].1.contains("Content two."));
}

#[test]
fn parse_note_pre_heading_content_gets_empty_heading() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("note.md");
    fs::write(
        &path,
        "Intro text before any heading.\n# First Section\nBody.\n",
    )
    .unwrap();

    let note = parse_note(&path, None).unwrap();

    assert!(!note.sections.is_empty());
    // First section should have empty heading key for pre-heading content
    assert_eq!(note.sections[0].0, "");
    assert!(note.sections[0].1.contains("Intro text"));
}

#[test]
fn parse_note_no_headings_single_section() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("note.md");
    fs::write(&path, "Just plain text.\nAnother line.\n").unwrap();

    let note = parse_note(&path, None).unwrap();

    assert_eq!(note.sections.len(), 1);
    assert_eq!(note.sections[0].0, "");
    assert!(note.sections[0].1.contains("Just plain text."));
}

// ---------------------------------------------------------------------------
// parse_note: body and content
// ---------------------------------------------------------------------------

#[test]
fn parse_note_body_excludes_frontmatter() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("note.md");
    let content = "---\nkey: value\n---\nBody content here.\n";
    fs::write(&path, content).unwrap();

    let note = parse_note(&path, None).unwrap();

    assert_eq!(note.content, content);
    assert_eq!(note.body, "Body content here.\n");
    assert!(!note.body.contains("---"));
}

#[test]
fn parse_note_body_is_full_content_when_no_frontmatter() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("note.md");
    let content = "Just body.\n";
    fs::write(&path, content).unwrap();

    let note = parse_note(&path, None).unwrap();

    assert_eq!(note.body, content);
    assert_eq!(note.content, content);
}

// ---------------------------------------------------------------------------
// parse_note: validation errors
// ---------------------------------------------------------------------------

#[test]
fn parse_note_error_nonexistent_file() {
    let result = parse_note(&PathBuf::from("/nonexistent/file.md"), None);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), ObsidianError::NotFound(_)));
}

#[test]
fn parse_note_error_file_too_large() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("huge.md");

    // Create a file that exceeds MAX_FILE_SIZE by writing sparse data
    let f = fs::File::create(&path).unwrap();
    f.set_len(MAX_FILE_SIZE + 1).unwrap();

    let result = parse_note(&path, None);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ObsidianError::FileTooLarge { .. }
    ));
}

#[test]
fn parse_note_error_outside_vault_root() {
    let vault = TempDir::new().unwrap();
    let outside = TempDir::new().unwrap();
    let path = outside.path().join("note.md");
    fs::write(&path, "content").unwrap();

    let result = parse_note(&path, Some(vault.path()));
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        ObsidianError::OutsideVault { .. }
    ));
}

#[test]
fn parse_note_ok_inside_vault_root() {
    let vault = TempDir::new().unwrap();
    let subdir = vault.path().join("notes");
    fs::create_dir(&subdir).unwrap();
    let path = subdir.join("note.md");
    fs::write(&path, "# Title\nBody.\n").unwrap();

    let note = parse_note(&path, Some(vault.path())).unwrap();
    assert_eq!(note.title, Some("Title".to_string()));
}

// ---------------------------------------------------------------------------
// parse_note: frontmatter integration
// ---------------------------------------------------------------------------

#[test]
fn parse_note_extracts_frontmatter() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("note.md");
    fs::write(
        &path,
        "---\ntags: review\ndifficulty: hard\n---\nBody.\n",
    )
    .unwrap();

    let note = parse_note(&path, None).unwrap();

    assert!(note.frontmatter.contains_key("tags"));
    assert!(note.frontmatter.contains_key("difficulty"));
}

// ---------------------------------------------------------------------------
// discover_notes: basic discovery
// ---------------------------------------------------------------------------

fn create_vault(dir: &TempDir) {
    // Top-level notes
    fs::write(dir.path().join("note1.md"), "# Note 1").unwrap();
    fs::write(dir.path().join("note2.md"), "# Note 2").unwrap();
    // Nested notes
    fs::create_dir(dir.path().join("subdir")).unwrap();
    fs::write(dir.path().join("subdir/note3.md"), "# Note 3").unwrap();
    // Non-md file
    fs::write(dir.path().join("readme.txt"), "not markdown").unwrap();
}

#[test]
fn discover_notes_finds_all_md_files() {
    let dir = TempDir::new().unwrap();
    create_vault(&dir);

    let notes = discover_notes(dir.path(), &["*.md"], DEFAULT_IGNORE_DIRS).unwrap();

    assert_eq!(notes.len(), 3);
    assert!(notes.iter().all(|p| p.extension().unwrap() == "md"));
}

#[test]
fn discover_notes_returns_sorted_paths() {
    let dir = TempDir::new().unwrap();
    create_vault(&dir);

    let notes = discover_notes(dir.path(), &["*.md"], DEFAULT_IGNORE_DIRS).unwrap();

    let is_sorted = notes.windows(2).all(|w| w[0] <= w[1]);
    assert!(is_sorted, "Paths should be sorted: {notes:?}");
}

#[test]
fn discover_notes_skips_default_ignore_dirs() {
    let dir = TempDir::new().unwrap();
    create_vault(&dir);

    // Create files in ignored directories
    for ignored in DEFAULT_IGNORE_DIRS {
        let ignored_dir = dir.path().join(ignored);
        fs::create_dir_all(&ignored_dir).unwrap();
        fs::write(ignored_dir.join("hidden.md"), "# Hidden").unwrap();
    }

    let notes = discover_notes(dir.path(), &["*.md"], DEFAULT_IGNORE_DIRS).unwrap();

    // Should only find the 3 vault notes, not the ones in ignored dirs
    assert_eq!(notes.len(), 3);
    for path in &notes {
        let path_str = path.to_string_lossy();
        for ignored in DEFAULT_IGNORE_DIRS {
            assert!(
                !path_str.contains(ignored),
                "Path {path_str} should not be in ignored dir {ignored}"
            );
        }
    }
}

#[test]
fn discover_notes_error_nonexistent_vault() {
    let result = discover_notes(
        &PathBuf::from("/nonexistent/vault"),
        &["*.md"],
        DEFAULT_IGNORE_DIRS,
    );
    assert!(result.is_err());
}

#[test]
fn discover_notes_skips_symlink_outside_vault() {
    let vault = TempDir::new().unwrap();
    let outside = TempDir::new().unwrap();

    // Real note inside vault
    fs::write(vault.path().join("real.md"), "# Real").unwrap();

    // Create a note outside the vault
    fs::write(outside.path().join("external.md"), "# External").unwrap();

    // Symlink from vault to outside file
    unix_fs::symlink(
        outside.path().join("external.md"),
        vault.path().join("link.md"),
    )
    .unwrap();

    let notes = discover_notes(vault.path(), &["*.md"], DEFAULT_IGNORE_DIRS).unwrap();

    // Should only find real.md, not the symlink to external
    assert_eq!(notes.len(), 1);
    assert!(notes[0].ends_with("real.md"));
}

#[test]
fn discover_notes_empty_vault() {
    let dir = TempDir::new().unwrap();

    let notes = discover_notes(dir.path(), &["*.md"], DEFAULT_IGNORE_DIRS).unwrap();

    assert!(notes.is_empty());
}

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tempfile::TempDir;

use super::*;
use crate::parser::ParsedNote;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// A mock generator that returns N cards per note.
struct FixedGenerator {
    cards_per_note: usize,
}

impl CardGenerator for FixedGenerator {
    fn generate(&self, _note: &ParsedNote) -> Vec<GeneratedCardRef> {
        (0..self.cards_per_note)
            .map(|i| GeneratedCardRef {
                slug: format!("card-{i}"),
                apf_html: format!("<p>Card {i}</p>"),
            })
            .collect()
    }
}

/// A mock generator that always returns an empty vec (no cards).
struct EmptyGenerator;

impl CardGenerator for EmptyGenerator {
    fn generate(&self, _note: &ParsedNote) -> Vec<GeneratedCardRef> {
        vec![]
    }
}

/// A mock generator that panics (simulates generation failure).
struct FailingGenerator;

impl CardGenerator for FailingGenerator {
    fn generate(&self, _note: &ParsedNote) -> Vec<GeneratedCardRef> {
        panic!("generator exploded")
    }
}

/// Helper to create a vault with N markdown files.
fn create_vault(dir: &TempDir, count: usize) -> PathBuf {
    let vault = dir.path().to_path_buf();
    for i in 0..count {
        let content = format!("---\ntitle: Note {i}\n---\n# Heading\nBody of note {i}\n");
        fs::write(vault.join(format!("note_{i}.md")), content).unwrap();
    }
    vault
}

/// Helper to create a vault with subdirectories.
fn create_vault_with_subdirs(dir: &TempDir) -> PathBuf {
    let vault = dir.path().to_path_buf();

    let topics = vault.join("topics");
    let archive = vault.join("archive");
    fs::create_dir_all(&topics).unwrap();
    fs::create_dir_all(&archive).unwrap();

    fs::write(
        topics.join("rust.md"),
        "---\ntitle: Rust\n---\n# Rust\nRust content\n",
    )
    .unwrap();
    fs::write(
        topics.join("python.md"),
        "---\ntitle: Python\n---\n# Python\nPython content\n",
    )
    .unwrap();
    fs::write(
        archive.join("old.md"),
        "---\ntitle: Old\n---\n# Old\nOld content\n",
    )
    .unwrap();
    fs::write(vault.join("root.md"), "# Root\nRoot content\n").unwrap();

    vault
}

// ---------------------------------------------------------------------------
// SyncResult::merge
// ---------------------------------------------------------------------------

#[test]
fn merge_combines_counts() {
    let a = SyncResult {
        generated: 3,
        updated: 1,
        skipped: 2,
        failed: 0,
        errors: vec![],
    };
    let b = SyncResult {
        generated: 5,
        updated: 2,
        skipped: 1,
        failed: 1,
        errors: vec![],
    };
    let merged = a.merge(b);
    assert_eq!(merged.generated, 8);
    assert_eq!(merged.updated, 3);
    assert_eq!(merged.skipped, 3);
    assert_eq!(merged.failed, 1);
}

#[test]
fn merge_combines_errors() {
    let a = SyncResult {
        errors: vec!["error A".to_string()],
        ..Default::default()
    };
    let b = SyncResult {
        errors: vec!["error B".to_string(), "error C".to_string()],
        ..Default::default()
    };
    let merged = a.merge(b);
    assert_eq!(merged.errors, vec!["error A", "error B", "error C"]);
}

#[test]
fn merge_with_default_is_identity() {
    let a = SyncResult {
        generated: 5,
        updated: 2,
        skipped: 1,
        failed: 3,
        errors: vec!["err".to_string()],
    };
    let merged = a.clone().merge(SyncResult::default());
    assert_eq!(merged.generated, 5);
    assert_eq!(merged.updated, 2);
    assert_eq!(merged.skipped, 1);
    assert_eq!(merged.failed, 3);
    assert_eq!(merged.errors, vec!["err"]);
}

#[test]
fn merge_two_defaults_is_default() {
    let merged = SyncResult::default().merge(SyncResult::default());
    assert_eq!(merged.generated, 0);
    assert_eq!(merged.updated, 0);
    assert_eq!(merged.skipped, 0);
    assert_eq!(merged.failed, 0);
    assert!(merged.errors.is_empty());
}

// ---------------------------------------------------------------------------
// ObsidianSyncWorkflow::new
// ---------------------------------------------------------------------------

#[test]
fn new_creates_workflow_without_progress() {
    let generator = FixedGenerator { cards_per_note: 1 };
    let _wf = ObsidianSyncWorkflow::new(generator, None);
}

#[test]
fn new_creates_workflow_with_progress() {
    let generator = FixedGenerator { cards_per_note: 1 };
    let cb: ProgressCallback = Box::new(|_phase, _current, _total| {});
    let _wf = ObsidianSyncWorkflow::new(generator, Some(cb));
}

// ---------------------------------------------------------------------------
// ObsidianSyncWorkflow::scan_vault
// ---------------------------------------------------------------------------

#[test]
fn scan_vault_discovers_all_notes() {
    let dir = TempDir::new().unwrap();
    let vault = create_vault(&dir, 3);
    let generator = FixedGenerator { cards_per_note: 1 };
    let wf = ObsidianSyncWorkflow::new(generator, None);

    let notes = wf.scan_vault(&vault, None);
    assert_eq!(notes.len(), 3);
}

#[test]
fn scan_vault_returns_parsed_notes_with_titles() {
    let dir = TempDir::new().unwrap();
    let vault = create_vault(&dir, 2);
    let generator = FixedGenerator { cards_per_note: 1 };
    let wf = ObsidianSyncWorkflow::new(generator, None);

    let notes = wf.scan_vault(&vault, None);
    // All notes have frontmatter title
    for note in &notes {
        assert!(note.title.is_some());
    }
}

#[test]
fn scan_vault_empty_vault_returns_empty() {
    let dir = TempDir::new().unwrap();
    let generator = FixedGenerator { cards_per_note: 1 };
    let wf = ObsidianSyncWorkflow::new(generator, None);

    let notes = wf.scan_vault(dir.path(), None);
    assert!(notes.is_empty());
}

#[test]
fn scan_vault_with_source_dirs_filters() {
    let dir = TempDir::new().unwrap();
    let vault = create_vault_with_subdirs(&dir);
    let generator = FixedGenerator { cards_per_note: 1 };
    let wf = ObsidianSyncWorkflow::new(generator, None);

    // Only scan "topics" subdirectory
    let notes = wf.scan_vault(&vault, Some(&["topics"]));
    assert_eq!(notes.len(), 2);
    for note in &notes {
        assert!(note.path.to_str().unwrap().contains("topics"));
    }
}

#[test]
fn scan_vault_with_multiple_source_dirs() {
    let dir = TempDir::new().unwrap();
    let vault = create_vault_with_subdirs(&dir);
    let generator = FixedGenerator { cards_per_note: 1 };
    let wf = ObsidianSyncWorkflow::new(generator, None);

    let notes = wf.scan_vault(&vault, Some(&["topics", "archive"]));
    assert_eq!(notes.len(), 3); // 2 in topics + 1 in archive
}

#[test]
fn scan_vault_source_dirs_nonexistent_subdir_returns_empty() {
    let dir = TempDir::new().unwrap();
    let vault = create_vault(&dir, 2);
    let generator = FixedGenerator { cards_per_note: 1 };
    let wf = ObsidianSyncWorkflow::new(generator, None);

    let notes = wf.scan_vault(&vault, Some(&["nonexistent"]));
    assert!(notes.is_empty());
}

#[test]
fn scan_vault_skips_unparseable_files() {
    let dir = TempDir::new().unwrap();
    let vault = dir.path().to_path_buf();
    // Write a valid note and a non-.md file
    fs::write(vault.join("good.md"), "# Good\nContent\n").unwrap();
    // Non-md files should be skipped by discover
    fs::write(vault.join("readme.txt"), "not a note").unwrap();
    let generator = FixedGenerator { cards_per_note: 1 };
    let wf = ObsidianSyncWorkflow::new(generator, None);

    let notes = wf.scan_vault(&vault, None);
    assert_eq!(notes.len(), 1);
}

// ---------------------------------------------------------------------------
// ObsidianSyncWorkflow::run
// ---------------------------------------------------------------------------

#[test]
fn run_generates_cards_for_all_notes() {
    let dir = TempDir::new().unwrap();
    let vault = create_vault(&dir, 3);
    let generator = FixedGenerator { cards_per_note: 2 };
    let wf = ObsidianSyncWorkflow::new(generator, None);

    let result = wf.run(&vault, None);
    assert_eq!(result.generated, 6); // 3 notes * 2 cards
    assert_eq!(result.failed, 0);
    assert!(result.errors.is_empty());
}

#[test]
fn run_empty_vault_returns_zero_counts() {
    let dir = TempDir::new().unwrap();
    let generator = FixedGenerator { cards_per_note: 1 };
    let wf = ObsidianSyncWorkflow::new(generator, None);

    let result = wf.run(dir.path(), None);
    assert_eq!(result.generated, 0);
    assert_eq!(result.failed, 0);
    assert_eq!(result.skipped, 0);
    assert!(result.errors.is_empty());
}

#[test]
fn run_empty_generator_counts_as_failed() {
    let dir = TempDir::new().unwrap();
    let vault = create_vault(&dir, 2);
    let generator = EmptyGenerator;
    let wf = ObsidianSyncWorkflow::new(generator, None);

    let result = wf.run(&vault, None);
    assert_eq!(result.generated, 0);
    assert_eq!(result.failed, 2); // Both notes produced zero cards
}

#[test]
fn run_respects_source_dirs() {
    let dir = TempDir::new().unwrap();
    let vault = create_vault_with_subdirs(&dir);
    let generator = FixedGenerator { cards_per_note: 1 };
    let wf = ObsidianSyncWorkflow::new(generator, None);

    let result = wf.run(&vault, Some(&["topics"]));
    assert_eq!(result.generated, 2); // Only 2 notes in topics/
}

#[test]
fn run_invokes_progress_callback() {
    let dir = TempDir::new().unwrap();
    let vault = create_vault(&dir, 2);
    let generator = FixedGenerator { cards_per_note: 1 };

    let call_count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&call_count);
    let cb: ProgressCallback = Box::new(move |_phase, _current, _total| {
        counter.fetch_add(1, Ordering::Relaxed);
    });

    let wf = ObsidianSyncWorkflow::new(generator, Some(cb));
    let _result = wf.run(&vault, None);
    assert!(call_count.load(Ordering::Relaxed) > 0);
}

// ---------------------------------------------------------------------------
// Send + Sync assertions
// ---------------------------------------------------------------------------

fn _assert_send<T: Send>() {}
fn _assert_sync<T: Sync>() {}

#[test]
fn sync_result_is_send_sync() {
    _assert_send::<SyncResult>();
    _assert_sync::<SyncResult>();
}

#[test]
fn note_result_is_send_sync() {
    _assert_send::<NoteResult>();
    _assert_sync::<NoteResult>();
}

#[test]
fn generated_card_ref_is_send_sync() {
    _assert_send::<GeneratedCardRef>();
    _assert_sync::<GeneratedCardRef>();
}

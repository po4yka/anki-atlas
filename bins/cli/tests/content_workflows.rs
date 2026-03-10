use std::fs;

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use tempfile::TempDir;

fn cmd() -> Command {
    cargo_bin_cmd!("anki-atlas")
}

#[test]
fn generate_previews_sections_from_markdown_note() {
    let dir = TempDir::new().unwrap();
    let note = dir.path().join("ownership.md");
    fs::write(
        &note,
        "---\n\
title: Rust Ownership\n\
---\n\
# Ownership\n\
Rules for moves.\n\
# Borrowing\n\
Shared and mutable references.\n",
    )
    .unwrap();

    cmd()
        .args(["generate", note.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("title: Rust Ownership"))
        .stdout(predicate::str::contains("estimated_cards: 2"))
        .stdout(predicate::str::contains(
            "sections: # Ownership, # Borrowing",
        ))
        .stdout(predicate::str::contains(
            "preview-only workflow: no cards were persisted",
        ));
}

#[test]
fn validate_reports_invalid_input_and_quality_scores() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("card.txt");
    fs::write(
        &input,
        "What does ownership enforce?\n---\n<script>alert('x')</script>\n---\nfoo/bar\nKotlin::Coroutines\n",
    )
    .unwrap();

    cmd()
        .args(["validate", input.to_str().unwrap(), "--quality"])
        .assert()
        .success()
        .stdout(predicate::str::contains("valid: false"))
        .stdout(predicate::str::contains("issues:"))
        .stdout(predicate::str::contains("quality: overall="));
}

#[test]
fn obsidian_sync_dry_run_scans_filtered_vault() {
    let dir = TempDir::new().unwrap();
    let rust_note = dir.path().join("topics/rust.md");
    let old_note = dir.path().join("archive/old.md");
    fs::create_dir_all(rust_note.parent().unwrap()).unwrap();
    fs::create_dir_all(old_note.parent().unwrap()).unwrap();

    fs::write(
        &rust_note,
        "---\n\
title: Rust\n\
---\n\
# Ownership\n\
See [[missing-note]].\n\
# Borrowing\n\
Borrowing rules.\n",
    )
    .unwrap();
    fs::write(
        &old_note,
        "---\n\
title: Old\n\
---\n\
# Archive\n\
Archived content.\n",
    )
    .unwrap();

    cmd()
        .args([
            "obsidian-sync",
            dir.path().to_str().unwrap(),
            "--source-dirs",
            "topics",
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("notes: 2"))
        .stdout(predicate::str::contains("generated_cards: 2"))
        .stdout(predicate::str::contains("orphaned_notes: 1"))
        .stdout(predicate::str::contains("broken_links: 1"))
        .stdout(predicate::str::contains("topics/rust.md"));
}

#[test]
fn tag_audit_fix_normalizes_and_rewrites_tags_file() {
    let dir = TempDir::new().unwrap();
    let tags = dir.path().join("tags.txt");
    fs::write(&tags, "Kotlin::Coroutines\nfoo/bar\ndifficulty::hard\n").unwrap();

    cmd()
        .args(["tag-audit", tags.to_str().unwrap(), "--fix"])
        .assert()
        .success()
        .stdout(predicate::str::contains("applied_fixes: true"))
        .stdout(predicate::str::contains(
            "Kotlin::Coroutines valid=false normalized=kotlin_coroutines",
        ))
        .stdout(predicate::str::contains(
            "foo/bar valid=false normalized=foo-bar",
        ));

    let rewritten = fs::read_to_string(&tags).unwrap();
    assert_eq!(rewritten, "difficulty::hard\nfoo-bar\nkotlin_coroutines\n");
}

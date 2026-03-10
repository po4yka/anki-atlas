use assert_cmd::Command;
use predicates::prelude::*;

fn cmd() -> Command {
    Command::cargo_bin("anki-atlas").expect("binary should exist")
}

#[test]
fn version_command_prints_version() {
    cmd()
        .arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains("anki-atlas"));
}

#[test]
fn no_args_shows_help_or_error() {
    cmd().assert().failure();
}

#[test]
fn help_flag_shows_usage() {
    cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Searchable hybrid index"));
}

#[test]
fn sync_help_shows_source_option() {
    cmd()
        .args(["sync", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--source"));
}

#[test]
fn search_help_shows_query_arg() {
    cmd()
        .args(["search", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("query"));
}

#[test]
fn migrate_command_runs() {
    // Should succeed (or at least not panic with todo!)
    cmd().arg("migrate").assert().success();
}

#[test]
fn sync_validates_source_exists() {
    cmd()
        .args(["sync", "--source", "/nonexistent/path.anki2"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found").or(predicate::str::contains("error")));
}

#[test]
fn generate_with_nonexistent_file_fails() {
    cmd()
        .args(["generate", "/nonexistent/note.md"])
        .assert()
        .failure();
}

#[test]
fn validate_with_nonexistent_file_fails() {
    cmd()
        .args(["validate", "/nonexistent/cards.txt"])
        .assert()
        .failure();
}

#[test]
fn tag_audit_with_nonexistent_file_fails() {
    cmd()
        .args(["tag-audit", "/nonexistent/tags.txt"])
        .assert()
        .failure();
}

#[test]
fn obsidian_sync_with_nonexistent_vault_fails() {
    cmd()
        .args(["obsidian-sync", "/nonexistent/vault"])
        .assert()
        .failure();
}

#[test]
fn index_command_runs() {
    cmd()
        .arg("index")
        .assert()
        .failure()
        .stderr(predicate::str::contains("not wired"));
}

#[test]
fn search_command_with_query_runs() {
    cmd()
        .args(["search", "test query"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not wired"));
}

#[test]
fn duplicates_command_runs() {
    cmd()
        .arg("duplicates")
        .assert()
        .failure()
        .stderr(predicate::str::contains("not wired"));
}

#[test]
fn topics_command_runs() {
    cmd()
        .arg("topics")
        .assert()
        .failure()
        .stderr(predicate::str::contains("not wired"));
}

#[test]
fn coverage_command_with_topic_runs() {
    cmd()
        .args(["coverage", "test/topic"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not wired"));
}

#[test]
fn gaps_command_with_topic_runs() {
    cmd()
        .args(["gaps", "test/topic"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not wired"));
}

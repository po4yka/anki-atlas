use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;

fn cmd() -> Command {
    cargo_bin_cmd!("anki-atlas")
}

#[test]
fn version_command_prints_version() {
    cmd()
        .arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains(env!("CARGO_PKG_VERSION")));
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
fn search_help_mentions_search_command() {
    cmd()
        .args(["search", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("search"));
}

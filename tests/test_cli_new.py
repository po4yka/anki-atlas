"""Tests for new CLI commands: generate, validate, obsidian-sync, tag-audit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

from apps.cli import app

runner = CliRunner()


class TestCommandRegistration:
    """Verify all new commands appear in the CLI."""

    def test_help_shows_generate(self) -> None:
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "Obsidian" in result.output

    def test_help_shows_validate(self) -> None:
        result = runner.invoke(app, ["validate", "--help"])
        assert result.exit_code == 0
        assert "flashcard" in result.output.lower() or "Validate" in result.output

    def test_help_shows_obsidian_sync(self) -> None:
        result = runner.invoke(app, ["obsidian-sync", "--help"])
        assert result.exit_code == 0
        assert "vault" in result.output.lower()

    def test_help_shows_tag_audit(self) -> None:
        result = runner.invoke(app, ["tag-audit", "--help"])
        assert result.exit_code == 0
        assert "tag" in result.output.lower()

    def test_existing_version_command(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestGenerateCommand:
    """Test the generate command."""

    def test_missing_file(self) -> None:
        result = runner.invoke(app, ["generate", "/nonexistent/file.md"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "Error" in result.output

    def test_valid_note(self, tmp_path: Path) -> None:
        note = tmp_path / "test.md"
        note.write_text("---\ntitle: Test Note\n---\n\n# Section 1\n\nSome content here.")
        result = runner.invoke(app, ["generate", str(note)])
        assert result.exit_code == 0
        assert "Test Note" in result.output

    def test_dry_run(self, tmp_path: Path) -> None:
        note = tmp_path / "test.md"
        note.write_text("# Hello\n\nWorld")
        result = runner.invoke(app, ["generate", str(note), "--dry-run"])
        assert result.exit_code == 0
        assert "Dry run" in result.output


class TestValidateCommand:
    """Test the validate command."""

    def test_missing_file(self) -> None:
        result = runner.invoke(app, ["validate", "/nonexistent/file.txt"])
        assert result.exit_code == 1

    def test_valid_card(self, tmp_path: Path) -> None:
        card_file = tmp_path / "cards.txt"
        card_file.write_text("What is the capital of France?\n---\nParis is the capital of France.")
        result = runner.invoke(app, ["validate", str(card_file)])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_empty_card_fails(self, tmp_path: Path) -> None:
        card_file = tmp_path / "cards.txt"
        card_file.write_text("Short\n---\n")
        result = runner.invoke(app, ["validate", str(card_file)])
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_quality_flag(self, tmp_path: Path) -> None:
        card_file = tmp_path / "cards.txt"
        card_file.write_text("What is the capital of France?\n---\nParis is the capital of France.")
        result = runner.invoke(app, ["validate", str(card_file), "--quality"])
        assert result.exit_code == 0
        assert "overall=" in result.output

    def test_no_cards_in_file(self, tmp_path: Path) -> None:
        card_file = tmp_path / "empty.txt"
        card_file.write_text("just some text without separator")
        result = runner.invoke(app, ["validate", str(card_file)])
        assert result.exit_code == 1
        assert "No cards found" in result.output


class TestObsidianSyncCommand:
    """Test the obsidian-sync command."""

    def test_missing_vault(self) -> None:
        result = runner.invoke(app, ["obsidian-sync", "/nonexistent/vault"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "Error" in result.output

    def test_dry_run_with_notes(self, tmp_path: Path) -> None:
        (tmp_path / "note1.md").write_text("# Note 1\n\nContent")
        (tmp_path / "note2.md").write_text("# Note 2\n\nContent")
        result = runner.invoke(app, ["obsidian-sync", str(tmp_path), "--dry-run"])
        assert result.exit_code == 0
        assert "2" in result.output
        assert "Dry run" in result.output

    def test_empty_vault(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["obsidian-sync", str(tmp_path), "--dry-run"])
        assert result.exit_code == 0
        assert "No notes found" in result.output

    def test_source_dirs(self, tmp_path: Path) -> None:
        subdir = tmp_path / "notes"
        subdir.mkdir()
        (subdir / "test.md").write_text("# Test\n\nContent")
        result = runner.invoke(
            app, ["obsidian-sync", str(tmp_path), "--source-dirs", "notes", "--dry-run"]
        )
        assert result.exit_code == 0
        assert "1" in result.output


class TestTagAuditCommand:
    """Test the tag-audit command."""

    def test_missing_file(self) -> None:
        result = runner.invoke(app, ["tag-audit", "/nonexistent/tags.txt"])
        assert result.exit_code == 1

    def test_valid_tags(self, tmp_path: Path) -> None:
        tag_file = tmp_path / "tags.txt"
        tag_file.write_text("kotlin::coroutines\nandroid::lifecycle\n")
        result = runner.invoke(app, ["tag-audit", str(tag_file)])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_invalid_tags(self, tmp_path: Path) -> None:
        tag_file = tmp_path / "tags.txt"
        tag_file.write_text("some--bad--tag\n")
        result = runner.invoke(app, ["tag-audit", str(tag_file)])
        assert result.exit_code == 0
        assert "FAIL" in result.output

    def test_fix_flag(self, tmp_path: Path) -> None:
        tag_file = tmp_path / "tags.txt"
        tag_file.write_text("kotlin::coroutines\n")
        result = runner.invoke(app, ["tag-audit", str(tag_file), "--fix"])
        assert result.exit_code == 0
        assert "Normalized" in result.output

    def test_empty_file(self, tmp_path: Path) -> None:
        tag_file = tmp_path / "empty.txt"
        tag_file.write_text("")
        result = runner.invoke(app, ["tag-audit", str(tag_file)])
        assert result.exit_code == 0
        assert "No tags found" in result.output

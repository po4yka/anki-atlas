"""Tests for packages.obsidian: frontmatter, parser, and analyzer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from packages.common.exceptions import ObsidianParseError
from packages.obsidian.analyzer import VaultAnalyzer, VaultStats
from packages.obsidian.frontmatter import (
    _preprocess_yaml_frontmatter,
    parse_frontmatter,
    write_frontmatter,
)
from packages.obsidian.parser import (
    MAX_FILE_SIZE,
    ParsedNote,
    _extract_title,
    _split_sections,
    discover_notes,
    parse_note,
)

# ---------------------------------------------------------------------------
# Frontmatter tests
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_valid_frontmatter(self) -> None:
        content = "---\ntitle: Hello\ntags: [a, b]\n---\nBody text"
        result = parse_frontmatter(content)
        assert result["title"] == "Hello"
        assert result["tags"] == ["a", "b"]

    def test_empty_frontmatter(self) -> None:
        content = "---\n---\nBody text"
        result = parse_frontmatter(content)
        assert result == {}

    def test_no_frontmatter(self) -> None:
        content = "Just some body text"
        result = parse_frontmatter(content)
        assert result == {}

    def test_invalid_yaml_raises(self) -> None:
        content = "---\n: invalid: yaml: [[\n---\nBody"
        with pytest.raises(ObsidianParseError, match="Invalid YAML"):
            parse_frontmatter(content)

    def test_frontmatter_with_nested_data(self) -> None:
        content = "---\nmeta:\n  key: value\n  count: 42\n---\nBody"
        result = parse_frontmatter(content)
        assert result["meta"] == {"key": "value", "count": 42}


class TestPreprocessFrontmatter:
    def test_removes_backticks(self) -> None:
        content = "---\ntitle: `Hello World`\n---\nBody"
        result = _preprocess_yaml_frontmatter(content)
        assert "`" not in result.split("---")[1]

    def test_no_frontmatter_passthrough(self) -> None:
        content = "No frontmatter here"
        assert _preprocess_yaml_frontmatter(content) == content


class TestWriteFrontmatter:
    def test_write_new_frontmatter(self) -> None:
        content = "Body text here"
        result = write_frontmatter({"title": "Test"}, content)
        assert result.startswith("---\n")
        assert "title: Test" in result
        assert result.endswith("---\nBody text here")

    def test_replace_existing_frontmatter(self) -> None:
        content = "---\nold: data\n---\nBody"
        result = write_frontmatter({"new": "data"}, content)
        assert "old: data" not in result
        assert "new: data" in result
        assert result.endswith("---\nBody")

    def test_roundtrip(self) -> None:
        data = {"title": "Hello", "tags": ["a", "b"]}
        content = write_frontmatter(data, "Body")
        parsed = parse_frontmatter(content)
        assert parsed["title"] == "Hello"
        assert parsed["tags"] == ["a", "b"]


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestExtractTitle:
    def test_title_from_frontmatter(self) -> None:
        assert _extract_title({"title": "FM Title"}, "# H1 Title") == "FM Title"

    def test_title_from_h1(self) -> None:
        assert _extract_title({}, "# My Heading\nSome text") == "My Heading"

    def test_no_title(self) -> None:
        assert _extract_title({}, "Just text") is None


class TestSplitSections:
    def test_no_headings(self) -> None:
        sections = _split_sections("Just plain text")
        assert sections == (("", "Just plain text"),)

    def test_single_heading(self) -> None:
        sections = _split_sections("# Heading\nContent here")
        assert len(sections) == 1
        assert sections[0] == ("# Heading", "Content here")

    def test_multiple_headings(self) -> None:
        body = "Intro\n\n# One\nA\n\n## Two\nB"
        sections = _split_sections(body)
        assert len(sections) == 3
        assert sections[0][0] == ""
        assert sections[1][0] == "# One"
        assert sections[2][0] == "## Two"

    def test_empty_body(self) -> None:
        assert _split_sections("") == ()


class TestParseNote:
    def test_parse_note_with_frontmatter(self, tmp_path: Path) -> None:
        note = tmp_path / "test.md"
        note.write_text("---\ntitle: Test Note\n---\n# Heading\nContent", encoding="utf-8")

        result = parse_note(note)
        assert isinstance(result, ParsedNote)
        assert result.frontmatter["title"] == "Test Note"
        assert result.title == "Test Note"
        assert result.body.startswith("# Heading")
        assert len(result.sections) >= 1

    def test_parse_note_without_frontmatter(self, tmp_path: Path) -> None:
        note = tmp_path / "bare.md"
        note.write_text("# Title\nBody content", encoding="utf-8")

        result = parse_note(note)
        assert result.frontmatter == {}
        assert result.title == "Title"

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ObsidianParseError, match="does not exist"):
            parse_note(tmp_path / "missing.md")

    def test_file_too_large(self, tmp_path: Path) -> None:
        big_file = tmp_path / "big.md"
        big_file.write_bytes(b"x" * (MAX_FILE_SIZE + 1))
        with pytest.raises(ObsidianParseError, match="too large"):
            parse_note(big_file)

    def test_symlink_traversal_blocked(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.md"
        secret.write_text("---\ntitle: Secret\n---\nHidden", encoding="utf-8")
        link = vault / "link.md"
        link.symlink_to(secret)

        with pytest.raises(ObsidianParseError, match="outside vault root"):
            parse_note(link, vault_root=vault)

    def test_symlink_inside_vault_ok(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        real = vault / "real.md"
        real.write_text("---\ntitle: Real\n---\nContent", encoding="utf-8")
        link = vault / "link.md"
        link.symlink_to(real)

        result = parse_note(link, vault_root=vault)
        assert result.title == "Real"

    def test_frozen_dataclass(self, tmp_path: Path) -> None:
        note = tmp_path / "test.md"
        note.write_text("---\ntitle: X\n---\nY", encoding="utf-8")
        result = parse_note(note)
        with pytest.raises(AttributeError):
            result.title = "Changed"  # type: ignore[misc]


class TestDiscoverNotes:
    def test_discover_md_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("note a", encoding="utf-8")
        (tmp_path / "b.md").write_text("note b", encoding="utf-8")
        (tmp_path / "c.txt").write_text("not a note", encoding="utf-8")

        notes = discover_notes(tmp_path)
        assert len(notes) == 2
        stems = {p.stem for p in notes}
        assert stems == {"a", "b"}

    def test_skip_ignored_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "good.md").write_text("good", encoding="utf-8")
        obsidian = tmp_path / ".obsidian"
        obsidian.mkdir()
        (obsidian / "config.md").write_text("config", encoding="utf-8")

        notes = discover_notes(tmp_path)
        assert len(notes) == 1
        assert notes[0].stem == "good"

    def test_custom_patterns(self, tmp_path: Path) -> None:
        (tmp_path / "q-test.md").write_text("q", encoding="utf-8")
        (tmp_path / "other.md").write_text("other", encoding="utf-8")

        notes = discover_notes(tmp_path, patterns=("q-*.md",))
        assert len(notes) == 1
        assert notes[0].stem == "q-test"

    def test_nonexistent_vault_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ObsidianParseError, match="not a directory"):
            discover_notes(tmp_path / "nonexistent")

    def test_symlink_outside_vault_skipped(self, tmp_path: Path) -> None:
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "real.md").write_text("real", encoding="utf-8")
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "external.md").write_text("external", encoding="utf-8")
        (vault / "link.md").symlink_to(outside / "external.md")

        notes = discover_notes(vault)
        assert len(notes) == 1
        assert notes[0].stem == "real"

    def test_sorted_output(self, tmp_path: Path) -> None:
        for name in ("c", "a", "b"):
            (tmp_path / f"{name}.md").write_text(name, encoding="utf-8")
        notes = discover_notes(tmp_path)
        stems = [p.stem for p in notes]
        assert stems == sorted(stems)


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------


class TestVaultAnalyzer:
    @pytest.fixture()
    def vault(self, tmp_path: Path) -> Path:
        v = tmp_path / "vault"
        v.mkdir()
        (v / "note-a.md").write_text(
            "---\ntitle: A\n---\nLinks to [[note-b]] and [[missing]]",
            encoding="utf-8",
        )
        (v / "note-b.md").write_text(
            "---\ntitle: B\n---\nLinks back to [[note-a]]",
            encoding="utf-8",
        )
        (v / "orphan.md").write_text("No links here", encoding="utf-8")
        return v

    def test_analyze_stats(self, vault: Path) -> None:
        analyzer = VaultAnalyzer(vault)
        stats = analyzer.analyze()

        assert isinstance(stats, VaultStats)
        assert stats.total_notes == 3
        assert stats.notes_with_frontmatter == 2
        assert stats.wikilinks_count == 3  # note-b, missing, note-a

    def test_broken_links(self, vault: Path) -> None:
        analyzer = VaultAnalyzer(vault)
        stats = analyzer.analyze()

        broken_targets = [target for _, target in stats.broken_links]
        assert "missing" in broken_targets

    def test_orphaned_notes(self, vault: Path) -> None:
        analyzer = VaultAnalyzer(vault)
        stats = analyzer.analyze()

        assert "orphan" in stats.orphaned_notes

    def test_get_wikilinks(self, vault: Path) -> None:
        analyzer = VaultAnalyzer(vault)
        links = analyzer.get_wikilinks(vault / "note-a.md")
        assert "note-b" in links
        assert "missing" in links

    def test_find_orphaned(self, vault: Path) -> None:
        analyzer = VaultAnalyzer(vault)
        orphaned = analyzer.find_orphaned()
        orphaned_stems = {p.stem for p in orphaned}
        assert "orphan" in orphaned_stems
        assert "note-a" not in orphaned_stems

    def test_frozen_vault_stats(self, vault: Path) -> None:
        analyzer = VaultAnalyzer(vault)
        stats = analyzer.analyze()
        with pytest.raises(AttributeError):
            stats.total_notes = 0  # type: ignore[misc]

    def test_wikilink_with_alias(self, tmp_path: Path) -> None:
        v = tmp_path / "vault"
        v.mkdir()
        (v / "test.md").write_text("Link: [[target|display text]]", encoding="utf-8")

        analyzer = VaultAnalyzer(v)
        links = analyzer.get_wikilinks(v / "test.md")
        assert links == ["target"]

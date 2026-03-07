"""Tests for packages.obsidian.sync."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from packages.generator.agents.models import GeneratedCard
from packages.obsidian.parser import ParsedNote
from packages.obsidian.sync import ObsidianSyncWorkflow, SyncResult
from packages.validation.pipeline import (
    Severity,
    ValidationIssue,
    ValidationPipeline,
    ValidationResult,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_note(path: Path, title: str = "Test") -> ParsedNote:
    return ParsedNote(
        path=path,
        frontmatter={},
        content=f"# {title}\nbody",
        body=f"# {title}\nbody",
        sections=(("", "body"),),
        title=title,
    )


def _make_card(slug: str = "test-card", index: int = 0) -> GeneratedCard:
    return GeneratedCard(
        card_index=index,
        slug=slug,
        lang="en",
        apf_html="<p>Q</p>",
    )


class FakeGenerator:
    """Generator that returns preset cards."""

    def __init__(self, cards: list[GeneratedCard] | None = None) -> None:
        self._cards = cards or [_make_card()]

    def generate(self, _note: ParsedNote) -> list[GeneratedCard]:
        return list(self._cards)


class FailingGenerator:
    """Generator that always raises."""

    def generate(self, _note: ParsedNote) -> list[GeneratedCard]:
        msg = "generation error"
        raise RuntimeError(msg)


# --- SyncResult ---


class TestSyncResult:
    def test_defaults(self) -> None:
        r = SyncResult()
        assert r.created == 0
        assert r.errors == ()

    def test_merge(self) -> None:
        a = SyncResult(created=1, failed=1, errors=("err1",))
        b = SyncResult(created=2, skipped=3, errors=("err2",))
        merged = a.merge(b)
        assert merged.created == 3
        assert merged.failed == 1
        assert merged.skipped == 3
        assert merged.errors == ("err1", "err2")


# --- scan_vault ---


class TestScanVault:
    def test_scan_discovers_notes(self, tmp_path: Path) -> None:
        (tmp_path / "note1.md").write_text("# Note 1\nbody")
        (tmp_path / "note2.md").write_text("# Note 2\nbody")
        wf = ObsidianSyncWorkflow(generator=FakeGenerator())
        notes = wf.scan_vault(tmp_path)
        assert len(notes) == 2
        titles = {n.title for n in notes}
        assert titles == {"Note 1", "Note 2"}

    def test_scan_with_source_dirs(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.md").write_text("# A\nbody")
        (tmp_path / "other.md").write_text("# Other\nbody")
        wf = ObsidianSyncWorkflow(generator=FakeGenerator())
        notes = wf.scan_vault(tmp_path, source_dirs=["src"])
        assert len(notes) == 1
        assert notes[0].title == "A"

    def test_scan_empty_vault(self, tmp_path: Path) -> None:
        wf = ObsidianSyncWorkflow(generator=FakeGenerator())
        notes = wf.scan_vault(tmp_path)
        assert notes == []

    def test_scan_reports_progress(self, tmp_path: Path) -> None:
        (tmp_path / "n.md").write_text("# N\nbody")
        progress: list[tuple[str, int, int]] = []
        wf = ObsidianSyncWorkflow(
            generator=FakeGenerator(),
            on_progress=lambda phase, cur, tot: progress.append((phase, cur, tot)),
        )
        wf.scan_vault(tmp_path)
        assert any(p[0] == "scan" for p in progress)


# --- process_note ---


class TestProcessNote:
    def test_process_returns_cards(self, tmp_path: Path) -> None:
        note = _make_note(tmp_path / "n.md")
        cards = [_make_card("c1"), _make_card("c2", index=1)]
        wf = ObsidianSyncWorkflow(generator=FakeGenerator(cards))
        result = wf.process_note(note)
        assert len(result.cards) == 2
        assert result.errors == []

    def test_process_generator_failure(self, tmp_path: Path) -> None:
        note = _make_note(tmp_path / "n.md")
        wf = ObsidianSyncWorkflow(generator=FailingGenerator())
        result = wf.process_note(note)
        assert result.cards == []
        assert len(result.errors) == 1
        assert "generation error" in result.errors[0]

    def test_process_with_validator_pass(self, tmp_path: Path) -> None:
        note = _make_note(tmp_path / "n.md")

        class PassValidator:
            def validate(self, *, front: str, back: str, tags: tuple[str, ...] = (), **_kw: Any) -> ValidationResult:  # noqa: ARG002
                return ValidationResult.ok()

        pipeline = ValidationPipeline([PassValidator()])
        wf = ObsidianSyncWorkflow(generator=FakeGenerator(), validator=pipeline)
        result = wf.process_note(note)
        assert len(result.cards) == 1
        assert result.errors == []

    def test_process_with_validator_fail(self, tmp_path: Path) -> None:
        note = _make_note(tmp_path / "n.md")

        class FailValidator:
            def validate(self, *, front: str, back: str, tags: tuple[str, ...] = (), **_kw: Any) -> ValidationResult:  # noqa: ARG002
                return ValidationResult(
                    issues=(ValidationIssue(severity=Severity.ERROR, message="bad card"),)
                )

        pipeline = ValidationPipeline([FailValidator()])
        wf = ObsidianSyncWorkflow(generator=FakeGenerator(), validator=pipeline)
        result = wf.process_note(note)
        assert result.cards == []
        assert len(result.errors) == 1
        assert "bad card" in result.errors[0]


# --- run ---


class TestRun:
    def test_run_full_pipeline(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("# A\nbody")
        (tmp_path / "b.md").write_text("# B\nbody")
        cards = [_make_card("c1")]
        wf = ObsidianSyncWorkflow(generator=FakeGenerator(cards))
        result = wf.run(tmp_path)
        assert result.created == 2  # 1 card per note, 2 notes
        assert result.failed == 0
        assert result.errors == ()

    def test_run_with_failures(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("# A\nbody")
        wf = ObsidianSyncWorkflow(generator=FailingGenerator())
        result = wf.run(tmp_path)
        assert result.created == 0
        assert result.failed == 1
        assert len(result.errors) == 1

    def test_run_reports_progress(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("# A\nbody")
        progress: list[tuple[str, int, int]] = []
        wf = ObsidianSyncWorkflow(
            generator=FakeGenerator(),
            on_progress=lambda phase, cur, tot: progress.append((phase, cur, tot)),
        )
        wf.run(tmp_path)
        phases = {p[0] for p in progress}
        assert "scan" in phases
        assert "process" in phases

    def test_run_empty_vault(self, tmp_path: Path) -> None:
        wf = ObsidianSyncWorkflow(generator=FakeGenerator())
        result = wf.run(tmp_path)
        assert result.created == 0
        assert result.failed == 0

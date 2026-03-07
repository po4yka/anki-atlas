"""Tests for new MCP tools: generate, validate, obsidian_sync, tag_audit."""

from __future__ import annotations

import pytest


class TestFormatGenerateResult:
    """Tests for format_generate_result."""

    def test_with_title_and_sections(self) -> None:
        from apps.mcp.formatters import format_generate_result

        output = format_generate_result(
            title="Calculus Basics",
            sections=(("## Derivatives", "Content about derivatives"),),
            body_length=200,
        )

        assert "Generation Preview" in output
        assert "Calculus Basics" in output
        assert "200 chars" in output
        assert "Sections**: 1" in output
        assert "Derivatives" in output

    def test_without_title(self) -> None:
        from apps.mcp.formatters import format_generate_result

        output = format_generate_result(
            title=None,
            sections=(),
            body_length=0,
        )

        assert "not detected" in output

    def test_estimated_cards(self) -> None:
        from apps.mcp.formatters import format_generate_result

        output = format_generate_result(
            title="Test",
            sections=(("## A", "a"), ("## B", "b"), ("## C", "c")),
            body_length=100,
        )

        assert "Estimated cards**: ~3" in output

    def test_empty_sections_estimates_one(self) -> None:
        from apps.mcp.formatters import format_generate_result

        output = format_generate_result(
            title="Test",
            sections=(),
            body_length=50,
        )

        assert "Estimated cards**: ~1" in output


class TestFormatValidateResult:
    """Tests for format_validate_result."""

    def test_passing_result(self) -> None:
        from apps.mcp.formatters import format_validate_result
        from packages.validation.pipeline import ValidationResult

        result = ValidationResult.ok()
        output = format_validate_result(result)

        assert "Validation: PASS" in output
        assert "Errors**: 0" in output

    def test_failing_result(self) -> None:
        from apps.mcp.formatters import format_validate_result
        from packages.validation.pipeline import Severity, ValidationIssue, ValidationResult

        result = ValidationResult(
            issues=(
                ValidationIssue(Severity.ERROR, "Front side is empty", "front"),
                ValidationIssue(Severity.WARNING, "Back side is very short", "back"),
            )
        )
        output = format_validate_result(result)

        assert "Validation: FAIL" in output
        assert "Errors**: 1" in output
        assert "Warnings**: 1" in output
        assert "Front side is empty" in output
        assert "(front)" in output

    def test_with_quality_score(self) -> None:
        from apps.mcp.formatters import format_validate_result
        from packages.validation.pipeline import ValidationResult
        from packages.validation.quality import QualityScore

        result = ValidationResult.ok()
        quality = QualityScore(
            clarity=0.9,
            atomicity=0.8,
            testability=0.7,
            memorability=0.85,
            accuracy=0.95,
        )
        output = format_validate_result(result, quality)

        assert "Quality Score" in output
        assert "Overall" in output
        assert "Clarity: 0.90" in output


class TestFormatObsidianSyncResult:
    """Tests for format_obsidian_sync_result."""

    def test_basic_result(self) -> None:
        from apps.mcp.formatters import format_obsidian_sync_result

        output = format_obsidian_sync_result(
            notes_found=5,
            parsed_notes=[
                ("note1.md", "First Note", 3),
                ("note2.md", None, 1),
            ],
            vault_path="/path/to/vault",
        )

        assert "Obsidian Vault Scan" in output
        assert "/path/to/vault" in output
        assert "Notes found**: 5" in output
        assert "First Note" in output
        assert "*(untitled)*" in output

    def test_empty_vault(self) -> None:
        from apps.mcp.formatters import format_obsidian_sync_result

        output = format_obsidian_sync_result(
            notes_found=0,
            parsed_notes=[],
            vault_path="/empty",
        )

        assert "Notes found**: 0" in output


class TestFormatTagAuditResult:
    """Tests for format_tag_audit_result."""

    def test_all_valid(self) -> None:
        from apps.mcp.formatters import format_tag_audit_result

        results = [
            ("math::calculus", [], None, []),
            ("cs::algorithms", [], None, []),
        ]
        output = format_tag_audit_result(results)

        assert "Total tags**: 2" in output
        assert "Valid**: 2" in output
        assert "All tags are valid" in output

    def test_with_issues(self) -> None:
        from apps.mcp.formatters import format_tag_audit_result

        results = [
            ("Math/calculus", ["Use '::' for hierarchy, not '/'"], "math-calculus", ["math::calculus"]),
            ("cs::algorithms", [], None, []),
        ]
        output = format_tag_audit_result(results)

        assert "With issues**: 1" in output
        assert "Issues Found" in output
        assert "Math/calculus" in output


class TestAnkiatlasValidateTool:
    """Tests for ankiatlas_validate tool."""

    @pytest.mark.asyncio
    async def test_valid_card(self) -> None:
        from apps.mcp.tools import ankiatlas_validate

        result = await ankiatlas_validate(
            front="What is the derivative of x^2?",
            back="The derivative of x^2 is 2x, found using the power rule.",
        )

        assert "PASS" in result
        assert "Quality Score" in result

    @pytest.mark.asyncio
    async def test_invalid_card(self) -> None:
        from apps.mcp.tools import ankiatlas_validate

        result = await ankiatlas_validate(
            front="",
            back="",
            check_quality=False,
        )

        assert "FAIL" in result
        assert "empty" in result.lower()

    @pytest.mark.asyncio
    async def test_with_tags(self) -> None:
        from apps.mcp.tools import ankiatlas_validate

        result = await ankiatlas_validate(
            front="What is recursion?",
            back="A function that calls itself to solve smaller subproblems.",
            tags=["cs::algorithms", "programming"],
        )

        assert "PASS" in result


class TestAnkiatlasTagAuditTool:
    """Tests for ankiatlas_tag_audit tool."""

    @pytest.mark.asyncio
    async def test_valid_tags(self) -> None:
        from apps.mcp.tools import ankiatlas_tag_audit

        result = await ankiatlas_tag_audit(
            tags=["math::calculus", "cs::algorithms"],
        )

        assert "Valid**: 2" in result

    @pytest.mark.asyncio
    async def test_invalid_tags_with_fix(self) -> None:
        from apps.mcp.tools import ankiatlas_tag_audit

        result = await ankiatlas_tag_audit(
            tags=["Math/calculus"],
            fix=True,
        )

        assert "With issues" in result

    @pytest.mark.asyncio
    async def test_empty_tag(self) -> None:
        from apps.mcp.tools import ankiatlas_tag_audit

        result = await ankiatlas_tag_audit(tags=["", "valid-tag"])

        assert "Total tags**: 2" in result


class TestAnkiatlasGenerateTool:
    """Tests for ankiatlas_generate tool."""

    @pytest.mark.asyncio
    async def test_simple_text(self) -> None:
        from apps.mcp.tools import ankiatlas_generate

        result = await ankiatlas_generate(
            text="# My Note\n\nSome content about a topic.\n\n## Section 1\n\nDetails here.",
        )

        assert "Generation Preview" in result
        assert "My Note" in result
        assert "Sections" in result

    @pytest.mark.asyncio
    async def test_empty_text(self) -> None:
        from apps.mcp.tools import ankiatlas_generate

        result = await ankiatlas_generate(text="")

        # Should still work (empty note)
        assert "Generation Preview" in result


class TestAnkiatlasObsidianSyncTool:
    """Tests for ankiatlas_obsidian_sync tool."""

    @pytest.mark.asyncio
    async def test_nonexistent_vault(self) -> None:
        from apps.mcp.tools import ankiatlas_obsidian_sync

        result = await ankiatlas_obsidian_sync(vault_path="/nonexistent/vault")

        assert "**Error**" in result
        assert "not found" in result.lower() or "not a directory" in result.lower()

    @pytest.mark.asyncio
    async def test_valid_vault(self, tmp_path: object) -> None:
        from pathlib import Path

        from apps.mcp.tools import ankiatlas_obsidian_sync

        vault = Path(str(tmp_path))
        (vault / "note1.md").write_text("# Test Note\n\nContent here.")
        (vault / "note2.md").write_text("# Another\n\n## Section\n\nMore content.")

        result = await ankiatlas_obsidian_sync(vault_path=str(vault))

        assert "Obsidian Vault Scan" in result
        assert "Notes found**: 2" in result
        assert "Test Note" in result
